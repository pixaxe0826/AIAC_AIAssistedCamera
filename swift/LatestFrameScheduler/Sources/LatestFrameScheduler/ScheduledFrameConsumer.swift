// Extracted from the private AI Camera Coach repository and adapted to run standalone.

public struct ScheduledFrameConsumer<Output: Sendable>: AnalysisFrameConsuming {
    private let scheduler: LatestFrameScheduler<Output>

    public init(scheduler: LatestFrameScheduler<Output>) {
        self.scheduler = scheduler
    }

    public func submit(frame: any AnalysisFrame) {
        Task { [scheduler] in await scheduler.submit(frame) }
    }
}
