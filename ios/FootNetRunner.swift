import Accelerate
import CoreML
import CoreVideo
import Foundation
import QuartzCore

// FootNet (research/foot-3d): a 256×256 crop of one foot in, 8 keypoint heatmaps out.
//
// Core ML directly rather than through ONNX Runtime: its Core ML execution provider cut this graph into 22 partitions
// and copied the tensors out and back at each one, which left the Neural Engine slower than the CPU. Core ML runs the
// whole network on the Neural Engine, 16× faster on the same weights.
enum FootNet {
  static let resource = "footnet"
  static let maskPreviewResource = "footmask-v1"
  static let size = FootNetDecoder.size
  static let joints = FootNetDecoder.joints
  static let maskPreviewSize = 64
  // The crop is this much wider than the box around the foot's points. Training saw 1.15 to 1.7 and validated at 1.3,
  // but on 94 real feet the wider crop is measurably better: 3.9 of 8 sure points against 3.0 at 1.3, and a fifth of
  // the feet with nothing at all instead of a third.
  static let context = 1.8
}

struct FootNetRun {
  let points: [Double]
  let maskPreviewMs: Double
}

final class FootNetRunner {
  private let model: MLModel
  private let maskPreviewModel: MLModel?
  private let pool: CVPixelBufferPool
  private let maskStatusLock = NSLock()
  private var maskStatusValue: String
  // The model writes float16; the peaks are found in float32, which Accelerate has the primitives for.
  private var heatmaps = [Float](repeating: 0, count: FootNet.joints * FootNet.size * FootNet.size)

  init(model: MLModel, maskPreviewModel: MLModel?, maskStatus: String) throws {
    self.model = model
    self.maskPreviewModel = maskPreviewModel
    maskStatusValue = maskStatus
    let attributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
      kCVPixelBufferWidthKey as String: FootNet.size,
      kCVPixelBufferHeightKey as String: FootNet.size,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
    ]
    var pool: CVPixelBufferPool?
    guard CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool) == kCVReturnSuccess, let pool else {
      throw FootPoseError.notReady("no pixel buffer pool for FootNet")
    }
    self.pool = pool
  }

  static func load(from bundle: Bundle) throws -> FootNetRunner {
    guard let url = bundle.url(forResource: FootNet.resource, withExtension: "mlmodelc") else {
      throw FootPoseError.modelMissing(FootNet.resource)
    }
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .all
    let maskPreviewModel: MLModel?
    let maskStatus: String
    if let maskURL = bundle.url(forResource: FootNet.maskPreviewResource, withExtension: "mlmodelc") {
      do {
        maskPreviewModel = try MLModel(contentsOf: maskURL, configuration: configuration)
        maskStatus = "ready"
      } catch {
        maskPreviewModel = nil
        maskStatus = "error loading model"
      }
    } else {
      maskPreviewModel = nil
      maskStatus = "missing"
    }
    return try FootNetRunner(
      model: try MLModel(contentsOf: url, configuration: configuration),
      maskPreviewModel: maskPreviewModel,
      maskStatus: maskStatus
    )
  }

  static func box(around points: [(x: Double, y: Double)]) -> Crop? {
    guard !points.isEmpty else { return nil }
    let xs = points.map(\.x)
    let ys = points.map(\.y)
    let side = max(xs.max()! - xs.min()!, ys.max()! - ys.min()!, 24) * FootNet.context
    return Crop(x: (xs.min()! + xs.max()!) / 2 - side / 2, y: (ys.min()! + ys.max()!) / 2 - side / 2, side: side)
  }

  /// The average brightness of the last crop, to tell a crop of a foot from one of nothing.
  private(set) var brightness: Double = 0

  var maskPreviewStatus: String {
    maskStatusLock.withLock { maskStatusValue }
  }

  /// Returns the existing points plus optional diagnostic mask timing.
  func run(_ pixelBuffer: CVPixelBuffer, crop box: Crop, maskPreview: Bool,
           threshold: Double, side: Int) throws -> FootNetRun {
    let input = try crop(pixelBuffer, to: box)
    brightness = Self.brightness(of: input)
    let output = try model.prediction(
      from: try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)])
    )
    guard let array = output.featureValue(for: "heatmaps")?.multiArrayValue,
          array.count == heatmaps.count else {
      throw FootPoseError.notReady("FootNet returned no heatmaps")
    }
    try read(array)
    let points = heatmaps.withUnsafeBufferPointer { FootNetDecoder.decode($0.baseAddress!, crop: box) }
    let maskPreviewMs = maskPreview
      ? runMaskPreview(input, crop: box, threshold: threshold, side: side,
                       frameWidth: CVPixelBufferGetWidth(pixelBuffer),
                       frameHeight: CVPixelBufferGetHeight(pixelBuffer))
      : 0
    return FootNetRun(points: points, maskPreviewMs: maskPreviewMs)
  }

  private func runMaskPreview(_ input: CVPixelBuffer, crop: Crop, threshold: Double, side: Int,
                              frameWidth: Int, frameHeight: Int) -> Double {
    guard let maskPreviewModel else { return 0 }
    let started = CACurrentMediaTime()
    do {
      let output = try maskPreviewModel.prediction(
        from: try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: input)])
      )
      guard let array = output.featureValue(for: "mask_logits")?.multiArrayValue,
            let pixels = try Self.previewPixels(array, threshold: threshold, side: side) else {
        setMaskStatus("error: no mask logits")
        return (CACurrentMediaTime() - started) * 1000
      }
      FootMaskPreviewStore.shared.set(
        FootMaskPreviewFrame(
          pixels: pixels,
          x: crop.x / Double(frameWidth),
          y: crop.y / Double(frameHeight),
          width: crop.side / Double(frameWidth),
          height: crop.side / Double(frameHeight)
        ),
        side: side
      )
      setMaskStatus("ready")
    } catch {
      setMaskStatus("error: inference failed")
    }
    return (CACurrentMediaTime() - started) * 1000
  }

  private func setMaskStatus(_ status: String) {
    maskStatusLock.withLock { maskStatusValue = status }
  }

  private static func previewPixels(_ array: MLMultiArray, threshold: Double, side: Int) throws -> Data? {
    let inputSize = FootNet.size
    let outputSize = FootNet.maskPreviewSize
    let inputCount = inputSize * inputSize
    guard array.dataType == .float16, array.count == inputCount else { return nil }
    var logits = [Float](repeating: 0, count: inputCount)
    try array.withUnsafeBytes { raw in
      try logits.withUnsafeMutableBufferPointer { floats in
        var source = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: raw.baseAddress!),
                                   height: 1, width: vImagePixelCount(inputCount), rowBytes: inputCount * 2)
        var destination = vImage_Buffer(data: floats.baseAddress!,
                                        height: 1, width: vImagePixelCount(inputCount), rowBytes: inputCount * 4)
        let converted = vImageConvert_Planar16FtoPlanarF(&source, &destination, vImage_Flags(kvImageNoFlags))
        guard converted == kvImageNoError else {
          throw FootPoseError.notReady("mask preview output conversion failed (\(converted))")
        }
      }
    }

    let alpha = 112.0
    let color: (UInt8, UInt8, UInt8) = side == 0 ? (27, 97, 58) : (112, 70, 4)
    let block = inputSize / outputSize
    let thresholdLogit: Float
    if threshold <= 0 {
      thresholdLogit = -.infinity
    } else if threshold >= 1 {
      thresholdLogit = .infinity
    } else {
      thresholdLogit = Float(log(threshold / (1 - threshold)))
    }
    var pixels = [UInt8](repeating: 0, count: outputSize * outputSize * 4)
    for y in 0..<outputSize {
      for x in 0..<outputSize {
        var visible = 0
        for dy in 0..<block {
          for dx in 0..<block {
            if logits[(y * block + dy) * inputSize + x * block + dx] >= thresholdLogit {
              visible += 1
            }
          }
        }
        guard visible > 0 else { continue }
        let coverage = Double(visible) / Double(block * block)
        let offset = (y * outputSize + x) * 4
        pixels[offset] = UInt8(Double(color.0) * coverage)
        pixels[offset + 1] = UInt8(Double(color.1) * coverage)
        pixels[offset + 2] = UInt8(Double(color.2) * coverage)
        pixels[offset + 3] = UInt8(alpha * coverage)
      }
    }
    return Data(pixels)
  }

  // Every 16th pixel is enough for an average, and it costs microseconds.
  private static func brightness(of buffer: CVPixelBuffer) -> Double {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to: UInt8.self) else { return 0 }
    let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    var sum = 0
    var count = 0
    for y in stride(from: 0, to: FootNet.size, by: 4) {
      for x in stride(from: 0, to: FootNet.size, by: 4) {
        let pixel = base + y * rowBytes + x * 4
        sum += Int(pixel[0]) + Int(pixel[1]) + Int(pixel[2])
        count += 3
      }
    }
    return Double(sum) / Double(count) / 255
  }

  private func read(_ array: MLMultiArray) throws {
    guard array.dataType == .float16 else {
      throw FootPoseError.notReady("FootNet heatmaps are \(array.dataType), expected float16")
    }
    let count = heatmaps.count
    try array.withUnsafeBytes { raw in
      heatmaps.withUnsafeMutableBufferPointer { floats in
        var source = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: raw.baseAddress!),
                                   height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
        var destination = vImage_Buffer(data: floats.baseAddress!,
                                        height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
        vImageConvert_Planar16FtoPlanarF(&source, &destination, vImage_Flags(kvImageNoFlags))
      }
    }
  }

  // The part of the crop inside the frame, scaled into the model's buffer; the rest stays black, as a replicated
  // border would lie anyway.
  private func crop(_ pixelBuffer: CVPixelBuffer, to box: Crop) throws -> CVPixelBuffer {
    guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else {
      throw FootPoseError.unsupportedFrame("expected BGRA, use pixelFormat 'rgb'")
    }
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
      throw FootPoseError.unsupportedFrame("no pixel buffer")
    }
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let left = max(0, Int(box.x.rounded()))
    let top = max(0, Int(box.y.rounded()))
    let right = min(width, Int((box.x + box.side).rounded()))
    let bottom = min(height, Int((box.y + box.side).rounded()))
    guard right - left > 8, bottom - top > 8 else {
      throw FootPoseError.unsupportedFrame("foot crop outside the frame")
    }
    let scale = Double(FootNet.size) / box.side
    // Where that part lands in the crop. Rounding the frame rect outwards makes this a fraction of a pixel negative,
    // which would put the scaled copy in front of the buffer, so it is clamped to the buffer.
    let insetX = min(FootNet.size - 1, max(0, Int((Double(left) - box.x) * scale)))
    let insetY = min(FootNet.size - 1, max(0, Int((Double(top) - box.y) * scale)))
    let scaledWidth = min(FootNet.size - insetX, Int((Double(right - left) * scale).rounded()))
    let scaledHeight = min(FootNet.size - insetY, Int((Double(bottom - top) * scale).rounded()))
    guard scaledWidth > 0, scaledHeight > 0 else {
      throw FootPoseError.unsupportedFrame("foot crop is empty")
    }

    var crop: CVPixelBuffer?
    guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &crop) == kCVReturnSuccess, let crop else {
      throw FootPoseError.notReady("no pixel buffer for the foot crop")
    }
    CVPixelBufferLockBaseAddress(crop, [])
    defer { CVPixelBufferUnlockBaseAddress(crop, []) }
    guard let cropBase = CVPixelBufferGetBaseAddress(crop) else {
      throw FootPoseError.notReady("no pixel buffer for the foot crop")
    }
    let cropRowBytes = CVPixelBufferGetBytesPerRow(crop)
    memset(cropBase, 0, cropRowBytes * FootNet.size)

    var source = vImage_Buffer(
      data: base.advanced(by: top * rowBytes + left * 4),
      height: vImagePixelCount(bottom - top),
      width: vImagePixelCount(right - left),
      rowBytes: rowBytes
    )
    var destination = vImage_Buffer(
      data: cropBase.advanced(by: insetY * cropRowBytes + insetX * 4),
      height: vImagePixelCount(scaledHeight),
      width: vImagePixelCount(scaledWidth),
      rowBytes: cropRowBytes
    )
    let scaled = vImageScale_ARGB8888(&source, &destination, nil, vImage_Flags(kvImageNoFlags))
    guard scaled == kvImageNoError else {
      throw FootPoseError.unsupportedFrame("vImage scale failed (\(scaled))")
    }
    return crop
  }
}
