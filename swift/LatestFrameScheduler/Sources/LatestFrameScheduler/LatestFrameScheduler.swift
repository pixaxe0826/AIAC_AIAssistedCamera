// Extracted from the private AI Camera Coach repository and adapted to run standalone.
import Foundation

/// Captures scheduler counters and retained-frame state at the time of inspection.
public struct LatestFrameSchedulerSnapshot: Equatable, Sendable {
    /// The number of submitted frames, including rejected submissions.
    public let submittedFrameCount: Int
    /// The number of frames passed to the processor.
    public let startedFrameCount: Int
    /// The number of accepted results passed to delivery.
    public let deliveredResultCount: Int
    /// The number of rejected submissions, rejected results, and inference failures.
    public let discardedResultCount: Int
    /// Whether the scheduler is draining queued work.
    public let hasInFlightFrame: Bool
    /// The identifier of the newest retained pending frame, if any.
    public let pendingFrameID: UInt64?
    /// The active camera generation, or nil before any frame is accepted.
    public let activeCameraGeneration: Int?
}

/// Serializes inference while retaining at most the latest pending frame.
public actor LatestFrameScheduler<Output: Sendable> {
    /// Processes one frame asynchronously and returns its result.
    public typealias Processor = @Sendable (any AnalysisFrame) async throws -> Output
    /// Decides whether a result is still valid against the latest scheduler state and time.
    public typealias Acceptance = @Sendable (
        _ output: Output,
        _ sourceFrame: any AnalysisFrame,
        _ newestFrameID: UInt64,
        _ activeCameraGeneration: Int,
        _ now: TimeInterval
    ) -> Bool
    /// Receives a result after it passes the acceptance check.
    public typealias Delivery = @Sendable (Output) -> Void

    private let processor: Processor
    private let acceptance: Acceptance
    private let delivery: Delivery
    private let timeSource: @Sendable () -> TimeInterval

    private var pendingFrame: (any AnalysisFrame)?
    private var isDraining = false
    private var newestFrameID: UInt64 = 0
    private var activeCameraGeneration = -1
    private var submittedFrameCount = 0
    private var startedFrameCount = 0
    private var deliveredResultCount = 0
    private var discardedResultCount = 0

    /// Creates a scheduler with caller-supplied processing, acceptance, clock, and delivery.
    public init(
        processor: @escaping Processor,
        acceptance: @escaping Acceptance,
        timeSource: @escaping @Sendable () -> TimeInterval = {
            ProcessInfo.processInfo.systemUptime
        },
        delivery: @escaping Delivery
    ) {
        self.processor = processor
        self.acceptance = acceptance
        self.timeSource = timeSource
        self.delivery = delivery
    }

    /// Retains the latest eligible frame and starts draining without awaiting inference.
    public func submit(_ frame: any AnalysisFrame) {
        submittedFrameCount += 1
        let generation = frame.metadata.cameraGeneration
        guard generation >= activeCameraGeneration else {
            discardedResultCount += 1
            return
        }
        if generation > activeCameraGeneration {
            activeCameraGeneration = generation
            newestFrameID = frame.metadata.frameID
            pendingFrame = nil
        } else {
            guard frame.metadata.frameID >= newestFrameID else {
                discardedResultCount += 1
                return
            }
            newestFrameID = frame.metadata.frameID
        }
        pendingFrame = frame
        guard !isDraining else { return }
        isDraining = true
        Task { await self.drain() }
    }

    /// Removes the pending frame without cancelling inference already in progress.
    public func cancelPending() {
        pendingFrame = nil
    }

    /// Returns the current counters, pending frame identifier, and camera generation.
    public func snapshot() -> LatestFrameSchedulerSnapshot {
        LatestFrameSchedulerSnapshot(
            submittedFrameCount: submittedFrameCount,
            startedFrameCount: startedFrameCount,
            deliveredResultCount: deliveredResultCount,
            discardedResultCount: discardedResultCount,
            hasInFlightFrame: isDraining,
            pendingFrameID: pendingFrame?.metadata.frameID,
            activeCameraGeneration: activeCameraGeneration >= 0 ? activeCameraGeneration : nil
        )
    }

    private func drain() async {
        while let frame = pendingFrame {
            pendingFrame = nil
            startedFrameCount += 1
            do {
                let output = try await processor(frame)
                if acceptance(
                    output,
                    frame,
                    newestFrameID,
                    activeCameraGeneration,
                    timeSource()
                ) {
                    deliveredResultCount += 1
                    delivery(output)
                } else {
                    discardedResultCount += 1
                }
            } catch {
                discardedResultCount += 1
            }
        }
        isDraining = false
    }
}

extension LatestFrameScheduler where Output: SchedulerRecommendation {
    /// Creates a scheduler that rejects mismatched, older-frame, older-generation, or expired results.
    public static func recommendations<Provider: CoachingInferenceProviding>(
        provider: Provider,
        timeSource: @escaping @Sendable () -> TimeInterval = {
            ProcessInfo.processInfo.systemUptime
        },
        delivery: @escaping Delivery
    ) -> LatestFrameScheduler<Output> where Provider.Output == Output {
        LatestFrameScheduler<Output>(
            processor: { frame in
                try await provider.recommend(frame: frame)
            },
            acceptance: { recommendation, frame, newestFrameID, generation, now in
                recommendation.contractVersion == .v1
                    && recommendation.sourceFrameID == frame.metadata.frameID
                    && recommendation.sourceFrameID == newestFrameID
                    && recommendation.sourceTimestamp == frame.metadata.timestamp
                    && recommendation.cameraGeneration == generation
                    && recommendation.cameraGeneration == frame.metadata.cameraGeneration
                    && recommendation.sourceTimestamp + recommendation.validFor >= now
            },
            timeSource: timeSource,
            delivery: delivery
        )
    }
}
