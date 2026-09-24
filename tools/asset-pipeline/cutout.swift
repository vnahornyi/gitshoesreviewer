import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

struct Options {
    var input: URL
    var outDir: URL
    var instance = "largest"
    var size = 1024
    var padding = 0.06
}

enum CutoutError: Error, CustomStringConvertible {
    case usage
    case unreadable(String)
    case noSubject(String)
    case badInstance(String, Int)
    case writeFailed(String)

    var description: String {
        switch self {
        case .usage:
            return "usage: swift cutout.swift <image> <out-dir> [--instance largest|all|<index>] [--size 1024] [--padding 0.06]"
        case .unreadable(let path): return "cannot read image: \(path)"
        case .noSubject(let path): return "no foreground subject found: \(path)"
        case .badInstance(let value, let count): return "instance \(value) out of range, found \(count)"
        case .writeFailed(let path): return "cannot write: \(path)"
        }
    }
}

func parseOptions() throws -> Options {
    var args = Array(CommandLine.arguments.dropFirst())
    guard args.count >= 2 else { throw CutoutError.usage }
    var options = Options(
        input: URL(fileURLWithPath: args.removeFirst()),
        outDir: URL(fileURLWithPath: args.removeFirst(), isDirectory: true)
    )
    while !args.isEmpty {
        let flag = args.removeFirst()
        guard !args.isEmpty else { throw CutoutError.usage }
        let value = args.removeFirst()
        switch flag {
        case "--instance": options.instance = value
        case "--size": options.size = Int(value) ?? options.size
        case "--padding": options.padding = Double(value) ?? options.padding
        default: throw CutoutError.usage
        }
    }
    return options
}

func loadImage(_ url: URL) throws -> CGImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else { throw CutoutError.unreadable(url.path) }
    return image
}

func squareCanvas(_ subject: CIImage, size: Int, padding: Double, context: CIContext) -> CGImage? {
    let extent = subject.extent
    let side = max(extent.width, extent.height) * (1 + 2 * padding)
    let scale = CGFloat(size) / side
    let scaled = subject
        .transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
        .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    let offsetX = (CGFloat(size) - extent.width * scale) / 2
    let offsetY = (CGFloat(size) - extent.height * scale) / 2
    let placed = scaled.transformed(by: CGAffineTransform(translationX: offsetX, y: offsetY))
    let canvas = CGRect(x: 0, y: 0, width: size, height: size)
    let composed = placed.composited(over: CIImage(color: .clear).cropped(to: canvas))
    return context.createCGImage(composed, from: canvas, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
}

func writePNG(_ image: CGImage, to url: URL) throws {
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw CutoutError.writeFailed(url.path) }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { throw CutoutError.writeFailed(url.path) }
}

func run() throws {
    let options = try parseOptions()
    let image = try loadImage(options.input)
    let handler = VNImageRequestHandler(cgImage: image)
    let request = VNGenerateForegroundInstanceMaskRequest()
    try handler.perform([request])

    guard let observation = request.results?.first, !observation.allInstances.isEmpty
    else { throw CutoutError.noSubject(options.input.path) }

    let context = CIContext()
    let instances = try observation.allInstances.sorted().map { index -> (index: Int, image: CIImage) in
        let buffer = try observation.generateMaskedImage(
            ofInstances: IndexSet(integer: index),
            from: handler,
            croppedToInstancesExtent: true
        )
        return (index, CIImage(cvPixelBuffer: buffer))
    }

    let selected: [(index: Int, image: CIImage)]
    switch options.instance {
    case "all":
        selected = instances
    case "largest":
        let largest = instances.max { $0.image.extent.width * $0.image.extent.height < $1.image.extent.width * $1.image.extent.height }
        selected = largest.map { [$0] } ?? []
    default:
        guard let position = Int(options.instance), instances.indices.contains(position)
        else { throw CutoutError.badInstance(options.instance, instances.count) }
        selected = [instances[position]]
    }

    try FileManager.default.createDirectory(at: options.outDir, withIntermediateDirectories: true)
    let stem = options.input.deletingPathExtension().lastPathComponent
    for (order, item) in selected.enumerated() {
        guard let square = squareCanvas(item.image, size: options.size, padding: options.padding, context: context)
        else { throw CutoutError.writeFailed(stem) }
        let url = options.outDir.appendingPathComponent("\(stem)-cutout-\(order).png")
        try writePNG(square, to: url)
        print("\(url.path)  instance=\(item.index)  extent=\(Int(item.image.extent.width))x\(Int(item.image.extent.height))")
    }
    print("instances found: \(instances.count)")
}

do {
    try run()
} catch {
    FileHandle.standardError.write("\(error)\n".data(using: .utf8)!)
    exit(1)
}
