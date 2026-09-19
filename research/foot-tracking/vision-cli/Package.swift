// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "FootVisionCLI",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "FootVisionCLI", path: "Sources/FootVisionCLI")
    ]
)
