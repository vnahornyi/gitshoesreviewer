import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

enum MaskKind: String, CaseIterable {
    case person
    case foreground
}

enum CLIError: Error, CustomStringConvertible {
    case usage
    case unreadable(String)
    case writeFailed(String)

    var description: String {
        switch self {
        case .usage:
            return "usage: FootVisionCLI <prepared-dir> <masks-dir> [--kind person|foreground|all]"
        case .unreadable(let path): return "cannot read image: \(path)"
        case .writeFailed(let path): return "cannot write: \(path)"
        }
    }
}

let context = CIContext()

func loadImage(_ url: URL) throws -> CGImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else { throw CLIError.unreadable(url.path) }
    return image
}

func personMask(_ image: CGImage) throws -> CIImage? {
    let request = VNGeneratePersonSegmentationRequest()
    request.qualityLevel = .accurate
    request.outputPixelFormat = kCVPixelFormatType_OneComponent8
    try VNImageRequestHandler(cgImage: image).perform([request])
    return request.results?.first.map { CIImage(cvPixelBuffer: $0.pixelBuffer) }
}

func foregroundMask(_ image: CGImage) throws -> CIImage? {
    let handler = VNImageRequestHandler(cgImage: image)
    let request = VNGenerateForegroundInstanceMaskRequest()
    try handler.perform([request])
    guard let observation = request.results?.first, !observation.allInstances.isEmpty else { return nil }
    let buffer = try observation.generateScaledMaskForImage(forInstances: observation.allInstances, from: handler)
    return CIImage(cvPixelBuffer: buffer)
}

func writeGrayPNG(_ mask: CIImage, width: Int, height: Int, to url: URL) throws {
    let scaled = mask.transformed(by: CGAffineTransform(
        scaleX: CGFloat(width) / mask.extent.width,
        y: CGFloat(height) / mask.extent.height
    ))
    let rect = CGRect(x: 0, y: 0, width: width, height: height)
    guard let rendered = context.createCGImage(scaled, from: rect, format: .L8, colorSpace: CGColorSpaceCreateDeviceGray()),
          let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw CLIError.writeFailed(url.path) }
    CGImageDestinationAddImage(destination, rendered, nil)
    guard CGImageDestinationFinalize(destination) else { throw CLIError.writeFailed(url.path) }
}

func run() throws {
    var args = Array(CommandLine.arguments.dropFirst())
    guard args.count >= 2 else { throw CLIError.usage }
    let inputDir = URL(fileURLWithPath: args.removeFirst(), isDirectory: true)
    let outputDir = URL(fileURLWithPath: args.removeFirst(), isDirectory: true)
    var kinds = MaskKind.allCases
    if args.count == 2, args[0] == "--kind" {
        kinds = args[1] == "all" ? MaskKind.allCases : [MaskKind(rawValue: args[1])].compactMap { $0 }
        guard !kinds.isEmpty else { throw CLIError.usage }
    } else if !args.isEmpty {
        throw CLIError.usage
    }

    let images = try FileManager.default
        .contentsOfDirectory(at: inputDir, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension.lowercased() == "jpg" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

    for kind in kinds {
        try FileManager.default.createDirectory(at: outputDir.appendingPathComponent(kind.rawValue), withIntermediateDirectories: true)
    }

    var missing: [MaskKind: Int] = [:]
    let started = Date()
    for url in images {
        let image = try loadImage(url)
        let stem = url.deletingPathExtension().lastPathComponent
        for kind in kinds {
            let mask = try kind == .person ? personMask(image) : foregroundMask(image)
            guard let mask else {
                missing[kind, default: 0] += 1
                continue
            }
            let target = outputDir.appendingPathComponent(kind.rawValue).appendingPathComponent("\(stem).png")
            try writeGrayPNG(mask, width: image.width, height: image.height, to: target)
        }
    }

    let elapsed = Date().timeIntervalSince(started)
    print("\(images.count) images, kinds: \(kinds.map(\.rawValue).joined(separator: ", ")), \(String(format: "%.1f", elapsed)) s")
    for (kind, count) in missing {
        print("no \(kind.rawValue) mask for \(count) images")
    }
}

do {
    try run()
} catch {
    FileHandle.standardError.write("\(error)\n".data(using: .utf8)!)
    exit(1)
}
