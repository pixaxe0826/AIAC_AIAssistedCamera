// Extracted from the private AI Camera Coach repository and adapted to run standalone.

/// Bridges synchronous camera-frame submission to the scheduler actor.
public struct ScheduledFrameConsumer<Output: Sendable>: AnalysisFrameConsuming {
    private let scheduler: LatestFrameScheduler<Output>

    /// Creates a consumer that forwards frames to the supplied scheduler.
    public init(scheduler: LatestFrameScheduler<Output>) {
        self.scheduler = scheduler
    }

    /// Schedules submission in a task so the caller does not wait for inference.
    public func submit(frame: any AnalysisFrame) {
        Task { [scheduler] in await scheduler.submit(frame) }
    }
}
