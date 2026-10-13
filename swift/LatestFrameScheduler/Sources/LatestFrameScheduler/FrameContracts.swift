// Extracted from the private AI Camera Coach repository and adapted to run standalone.
import Foundation

public enum AIContractVersion: String, Sendable {
    case v1 = "1"
}

public protocol FrameMetadata: Sendable {
    var frameID: UInt64 { get }
    var timestamp: TimeInterval { get }
    var cameraGeneration: Int { get }
}

public protocol AnalysisFrame: Sendable {
    var metadata: any FrameMetadata { get }
}

public protocol SchedulerRecommendation: Sendable {
    var contractVersion: AIContractVersion { get }
    var sourceFrameID: UInt64 { get }
    var sourceTimestamp: TimeInterval { get }
    var cameraGeneration: Int { get }
    var validFor: TimeInterval { get }
}

public protocol CoachingInferenceProviding: Sendable {
    associatedtype Output: SchedulerRecommendation
    func recommend(frame: any AnalysisFrame) async throws -> Output
}

public protocol AnalysisFrameConsuming: Sendable {
    func submit(frame: any AnalysisFrame)
}
