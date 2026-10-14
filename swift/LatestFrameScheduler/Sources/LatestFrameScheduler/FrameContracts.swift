// Extracted from the private AI Camera Coach repository and adapted to run standalone.
import Foundation

/// Identifies the recommendation contract used for result acceptance.
public enum AIContractVersion: String, Sendable {
    /// The supported version-one contract.
    case v1 = "1"
}

/// Supplies the identity, capture time, and camera generation of a frame.
public protocol FrameMetadata: Sendable {
    /// The frame identifier used to reject results from older frames.
    var frameID: UInt64 { get }
    /// The capture timestamp in the same time domain as the scheduler clock.
    var timestamp: TimeInterval { get }
    /// The camera generation used to reject frames and results from older sessions.
    var cameraGeneration: Int { get }
}

/// Represents a sendable analysis frame without requiring camera-framework types.
public protocol AnalysisFrame: Sendable {
    /// The identity and timing information attached to this frame.
    var metadata: any FrameMetadata { get }
}

/// Describes a result's source frame and validity window for stale-result rejection.
public protocol SchedulerRecommendation: Sendable {
    /// The contract version used to interpret this recommendation.
    var contractVersion: AIContractVersion { get }
    /// The identifier of the frame that produced this recommendation.
    var sourceFrameID: UInt64 { get }
    /// The capture timestamp of the frame that produced this recommendation.
    var sourceTimestamp: TimeInterval { get }
    /// The camera generation that produced this recommendation.
    var cameraGeneration: Int { get }
    /// The validity duration measured from the source timestamp.
    var validFor: TimeInterval { get }
}

/// Provides asynchronous inference for the recommendation scheduler.
public protocol CoachingInferenceProviding: Sendable {
    /// The recommendation type returned by this provider.
    associatedtype Output: SchedulerRecommendation
    /// Produces a recommendation for one analysis frame, or throws on inference failure.
    func recommend(frame: any AnalysisFrame) async throws -> Output
}

/// Accepts camera frames through a synchronous submission interface.
public protocol AnalysisFrameConsuming: Sendable {
    /// Submits a frame without exposing an asynchronous inference operation to the caller.
    func submit(frame: any AnalysisFrame)
}
