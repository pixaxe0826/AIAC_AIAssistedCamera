// Extracted from the private AI Camera Coach repository and adapted to run standalone.
import Foundation
import Testing
@testable import LatestFrameScheduler

struct AIInferenceSchedulerTests {
    @Test func keepsOneInferenceInFlightAndRetainsLatestPendingFrame() async throws {
        let gate = SchedulerTestGate()
        let scheduler = LatestFrameScheduler<UInt64>(
            processor: { frame in
                await gate.startAndWait(frame.metadata.frameID)
                return frame.metadata.frameID
            },
            acceptance: { output, frame, newestFrameID, generation, _ in
                output == frame.metadata.frameID
                    && output == newestFrameID
                    && frame.metadata.cameraGeneration == generation
            },
            timeSource: { 100.1 },
            delivery: { output in
                Task { await gate.deliver(output) }
            }
        )

        await scheduler.submit(try makeSchedulerFrame(frameID: 100))
        await waitUntil { await gate.startedFrames().count == 1 }
        await scheduler.submit(try makeSchedulerFrame(frameID: 101))
        await scheduler.submit(try makeSchedulerFrame(frameID: 102))
        await scheduler.submit(try makeSchedulerFrame(frameID: 103))

        var snapshot = await scheduler.snapshot()
        #expect(snapshot.hasInFlightFrame)
        #expect(snapshot.pendingFrameID == 103)
        await gate.releaseNext()
        await waitUntil { await gate.startedFrames().count == 2 }
        let startedFrames = await gate.startedFrames()
        #expect(startedFrames == [100, 103])
        await gate.releaseNext()
        await waitUntil { !(await scheduler.snapshot().hasInFlightFrame) }
        await waitUntil { await gate.deliveredFrames().count == 1 }

        snapshot = await scheduler.snapshot()
        #expect(snapshot.startedFrameCount == 2)
        #expect(snapshot.deliveredResultCount == 1)
        let deliveredFrames = await gate.deliveredFrames()
        #expect(deliveredFrames == [103])
    }

    @Test func recommendationSchedulerDiscardsExpiredResult() async throws {
        let provider = SchedulerRecommendationProvider(validFor: 0.05)
        let delivered = DeliveryRecorder()
        let scheduler = LatestFrameScheduler<TestRecommendation>.recommendations(
            provider: provider,
            timeSource: { 101 },
            delivery: { value in Task { await delivered.append(value.sourceFrameID) } }
        )
        await scheduler.submit(try makeSchedulerFrame(frameID: 1, timestamp: 100))
        await waitUntil { !(await scheduler.snapshot().hasInFlightFrame) }
        let deliveredValues = await delivered.values()
        let snapshot = await scheduler.snapshot()
        #expect(deliveredValues.isEmpty)
        #expect(snapshot.discardedResultCount == 1)
    }

    @Test func olderCameraGenerationCannotReplaceActiveGeneration() async throws {
        let recorder = DeliveryRecorder()
        let scheduler = LatestFrameScheduler<UInt64>(
            processor: { $0.metadata.frameID },
            acceptance: { output, frame, newest, generation, _ in
                output == newest && frame.metadata.cameraGeneration == generation
            },
            delivery: { value in Task { await recorder.append(value) } }
        )
        await scheduler.submit(try makeSchedulerFrame(frameID: 2, cameraGeneration: 2))
        await waitUntil { !(await scheduler.snapshot().hasInFlightFrame) }
        await scheduler.submit(try makeSchedulerFrame(frameID: 99, cameraGeneration: 1))
        await waitUntil { await scheduler.snapshot().submittedFrameCount == 2 }
        await waitUntil { await recorder.values().count == 1 }
        let snapshot = await scheduler.snapshot()
        let deliveredValues = await recorder.values()
        #expect(snapshot.activeCameraGeneration == 2)
        #expect(deliveredValues == [2])
    }

    @Test func outOfOrderFrameCannotReplaceNewerPendingFrame() async throws {
        let gate = SchedulerTestGate()
        let scheduler = LatestFrameScheduler<UInt64>(
            processor: { frame in
                await gate.startAndWait(frame.metadata.frameID)
                return frame.metadata.frameID
            },
            acceptance: { output, _, newest, _, _ in output == newest },
            delivery: { value in Task { await gate.deliver(value) } }
        )
        await scheduler.submit(try makeSchedulerFrame(frameID: 10))
        await waitUntil { await gate.startedFrames().count == 1 }
        await scheduler.submit(try makeSchedulerFrame(frameID: 12))
        await scheduler.submit(try makeSchedulerFrame(frameID: 11))
        let pendingSnapshot = await scheduler.snapshot()
        #expect(pendingSnapshot.pendingFrameID == 12)
        await gate.releaseNext()
        await waitUntil { await gate.startedFrames().count == 2 }
        await gate.releaseNext()
        await waitUntil { !(await scheduler.snapshot().hasInFlightFrame) }
        let started = await gate.startedFrames()
        #expect(started == [10, 12])
    }
    @Test func generationChangeDiscardsAlreadyRunningResult() async throws {
        let gate = SchedulerTestGate()
        let scheduler = LatestFrameScheduler<UInt64>(
            processor: { frame in
                await gate.startAndWait(frame.metadata.frameID)
                return frame.metadata.frameID
            },
            acceptance: { output, frame, newest, generation, _ in
                output == newest && frame.metadata.cameraGeneration == generation
            },
            delivery: { value in Task { await gate.deliver(value) } }
        )
        await scheduler.submit(try makeSchedulerFrame(frameID: 100, cameraGeneration: 1))
        await waitUntil { await gate.startedFrames().count == 1 }
        await scheduler.submit(try makeSchedulerFrame(frameID: 1, cameraGeneration: 2))
        await gate.releaseNext()
        await waitUntil { await gate.startedFrames().count == 2 }
        await gate.releaseNext()
        await waitUntil { !(await scheduler.snapshot().hasInFlightFrame) }
        await waitUntil { await gate.deliveredFrames().count == 1 }
        let delivered = await gate.deliveredFrames()
        #expect(delivered == [1])
        let snapshot = await scheduler.snapshot()
        #expect(snapshot.discardedResultCount == 1)
    }

    @Test func cameraSubmissionReturnsWhileInferenceIsSuspended() async throws {
        let gate = SchedulerTestGate()
        let scheduler = LatestFrameScheduler<UInt64>(
            processor: { frame in
                await gate.startAndWait(frame.metadata.frameID)
                return frame.metadata.frameID
            },
            acceptance: { output, _, newest, _, _ in output == newest },
            delivery: { value in Task { await gate.deliver(value) } }
        )
        let consumer = ScheduledFrameConsumer(scheduler: scheduler)
        consumer.submit(frame: try makeSchedulerFrame(frameID: 1))
        await waitUntil { await gate.startedFrames().count == 1 }
        var captureActions = 0
        consumer.submit(frame: try makeSchedulerFrame(frameID: 2))
        captureActions += 1
        #expect(captureActions == 1)
        let startedBeforeRelease = await gate.startedFrames()
        #expect(startedBeforeRelease == [1])
        await waitUntil { await scheduler.snapshot().pendingFrameID == 2 }
        await gate.releaseNext()
        await waitUntil { await gate.startedFrames().count == 2 }
        await gate.releaseNext()
        await waitUntil { !(await scheduler.snapshot().hasInFlightFrame) }
    }

}

private actor SchedulerTestGate {
    private var started: [UInt64] = []
    private var delivered: [UInt64] = []
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func startAndWait(_ frameID: UInt64) async {
        started.append(frameID)
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func releaseNext() {
        guard !waiters.isEmpty else { return }
        waiters.removeFirst().resume()
    }

    func deliver(_ frameID: UInt64) { delivered.append(frameID) }
    func startedFrames() -> [UInt64] { started }
    func deliveredFrames() -> [UInt64] { delivered }
}

private actor DeliveryRecorder {
    private var recorded: [UInt64] = []
    func append(_ value: UInt64) { recorded.append(value) }
    func values() -> [UInt64] { recorded }
}

private struct SchedulerRecommendationProvider: CoachingInferenceProviding {
    let validFor: TimeInterval

    func recommend(frame: any AnalysisFrame) async throws -> TestRecommendation {
        TestRecommendation(
            contractVersion: .v1,
            sourceFrameID: frame.metadata.frameID,
            sourceTimestamp: frame.metadata.timestamp,
            cameraGeneration: frame.metadata.cameraGeneration,
            validFor: validFor
        )
    }
}

private struct TestRecommendation: SchedulerRecommendation {
    let contractVersion: AIContractVersion
    let sourceFrameID: UInt64
    let sourceTimestamp: TimeInterval
    let cameraGeneration: Int
    let validFor: TimeInterval
}

private struct TestFrameMetadata: FrameMetadata {
    let frameID: UInt64
    let timestamp: TimeInterval
    let cameraGeneration: Int
}

private struct TestFrame: AnalysisFrame {
    let metadata: any FrameMetadata
}

private func makeSchedulerFrame(
    frameID: UInt64,
    timestamp: TimeInterval = 100,
    cameraGeneration: Int = 1
) throws -> TestFrame {
    TestFrame(metadata: TestFrameMetadata(frameID: frameID, timestamp: timestamp, cameraGeneration: cameraGeneration))
}

private func waitUntil(
    attempts: Int = 1_000,
    condition: @escaping () async -> Bool
) async {
    for _ in 0..<attempts {
        if await condition() { return }
        await Task.yield()
    }
    Issue.record("Timed out waiting for asynchronous scheduler state")
}
