// swift-tools-version: 5.9
// Extracted from the private AI Camera Coach repository and adapted to run standalone.
import PackageDescription

let package = Package(
    name: "LatestFrameScheduler",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "LatestFrameScheduler", targets: ["LatestFrameScheduler"])
    ],
    targets: [
        .target(name: "LatestFrameScheduler"),
        .testTarget(name: "LatestFrameSchedulerTests", dependencies: ["LatestFrameScheduler"])
    ],
    swiftLanguageVersions: [.v5]
)
