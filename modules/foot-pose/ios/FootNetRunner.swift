import Accelerate
import CoreVideo
import Foundation
import onnxruntime_objc

// FootNet (research/foot-3d): a 256×256 crop of one foot in, a foot mask and 8 keypoint heatmaps out.
enum FootNet {
  static let resource = "footnet-fp16"
  static let size = 256
  static let joints = 8
  // ImageNet normalisation on RGB 0…1, as the training pipeline uses.
  static let mean: [Float] = [0.485, 0.456, 0.406]
  static let std: [Float] = [0.229, 0.224, 0.225]
  // The crop is this much wider than the box around the points RTMPose found, matching the training crops.
  static let context = 1.45
  // Peaks are refined over this radius, as `decode_points` does in training.
  static let refineRadius = 2
}

// The crop's placement in the frame, to map the points back.
struct Crop {
  let x: Double
  let y: Double
  let side: Double
}

final class FootNetRunner {
  private let session: ORTSession
  private let inputData = NSMutableData(length: 3 * FootNet.size * FootNet.size * MemoryLayout<Float>.size)!
  private var crop = [UInt8](repeating: 0, count: FootNet.size * FootNet.size * 4)

  init(session: ORTSession) {
    self.session = session
  }

  static func box(around points: [(x: Double, y: Double)]) -> Crop? {
    guard !points.isEmpty else { return nil }
    let xs = points.map(\.x)
    let ys = points.map(\.y)
    let side = max(xs.max()! - xs.min()!, ys.max()! - ys.min()!, 24) * FootNet.context
    return Crop(x: (xs.min()! + xs.max()!) / 2 - side / 2, y: (ys.min()! + ys.max()!) / 2 - side / 2, side: side)
  }

  /// Returns 8 × [x, y, score] in frame pixels.
  func run(_ pixelBuffer: CVPixelBuffer, crop box: Crop) throws -> [Double] {
    try fillInput(from: pixelBuffer, crop: box)
    let outputs = try session.run(
      withInputs: ["image": try tensor(inputData, shape: [1, 3, FootNet.size, FootNet.size])],
      outputNames: ["heatmaps"],
      runOptions: nil
    )
    guard let heatmaps = try outputs["heatmaps"]?.tensorData() as Data? else { return [] }
    return heatmaps.withUnsafeBytes { raw in
      decode(raw.bindMemory(to: Float.self).baseAddress!, crop: box)
    }
  }

  private func tensor(_ data: NSMutableData, shape: [Int]) throws -> ORTValue {
    try ORTValue(tensorData: data, elementType: .float, shape: shape.map { NSNumber(value: $0) })
  }

  private func fillInput(from pixelBuffer: CVPixelBuffer, crop box: Crop) throws {
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
    // The part of the crop that is inside the frame; the rest stays black, as a replicated border would lie anyway.
    let left = max(0, Int(box.x.rounded()))
    let top = max(0, Int(box.y.rounded()))
    let right = min(width, Int((box.x + box.side).rounded()))
    let bottom = min(height, Int((box.y + box.side).rounded()))
    guard right - left > 8, bottom - top > 8 else {
      throw FootPoseError.unsupportedFrame("foot crop outside the frame")
    }
    let scale = Double(FootNet.size) / box.side
    let insetX = Int((Double(left) - box.x) * scale)
    let insetY = Int((Double(top) - box.y) * scale)
    let scaledWidth = min(FootNet.size - insetX, Int((Double(right - left) * scale).rounded()))
    let scaledHeight = min(FootNet.size - insetY, Int((Double(bottom - top) * scale).rounded()))
    guard scaledWidth > 0, scaledHeight > 0 else {
      throw FootPoseError.unsupportedFrame("foot crop is empty")
    }

    var source = vImage_Buffer(
      data: base.advanced(by: top * rowBytes + left * 4),
      height: vImagePixelCount(bottom - top),
      width: vImagePixelCount(right - left),
      rowBytes: rowBytes
    )
    let cropRowBytes = FootNet.size * 4
    let scaled = crop.withUnsafeMutableBytes { bytes -> vImage_Error in
      bytes.initializeMemory(as: UInt8.self, repeating: 0)
      var destination = vImage_Buffer(
        data: bytes.baseAddress! + insetY * cropRowBytes + insetX * 4,
        height: vImagePixelCount(scaledHeight),
        width: vImagePixelCount(scaledWidth),
        rowBytes: cropRowBytes
      )
      return vImageScale_ARGB8888(&source, &destination, nil, vImage_Flags(kvImageNoFlags))
    }
    guard scaled == kvImageNoError else {
      throw FootPoseError.unsupportedFrame("vImage scale failed (\(scaled))")
    }

    let plane = FootNet.size * FootNet.size
    let input = inputData.mutableBytes.assumingMemoryBound(to: Float.self)
    crop.withUnsafeBufferPointer { pixels in
      for index in 0..<plane {
        let blue = Float(pixels[index * 4]) / 255
        let green = Float(pixels[index * 4 + 1]) / 255
        let red = Float(pixels[index * 4 + 2]) / 255
        input[index] = (red - FootNet.mean[0]) / FootNet.std[0]
        input[plane + index] = (green - FootNet.mean[1]) / FootNet.std[1]
        input[2 * plane + index] = (blue - FootNet.mean[2]) / FootNet.std[2]
      }
    }
  }

  // Peak of each heatmap, refined by a softmax-weighted mean over its neighbourhood, then mapped back to the frame.
  private func decode(_ heatmaps: UnsafePointer<Float>, crop box: Crop) -> [Double] {
    let plane = FootNet.size * FootNet.size
    return (0..<FootNet.joints).flatMap { joint -> [Double] in
      let map = heatmaps + joint * plane
      var peak: Float = 0
      var index: vDSP_Length = 0
      vDSP_maxvi(map, 1, &peak, &index, vDSP_Length(plane))
      let peakX = Int(index) % FootNet.size
      let peakY = Int(index) / FootNet.size

      var weight = 0.0
      var x = 0.0
      var y = 0.0
      for dy in -FootNet.refineRadius...FootNet.refineRadius {
        for dx in -FootNet.refineRadius...FootNet.refineRadius {
          let sampleX = min(max(peakX + dx, 0), FootNet.size - 1)
          let sampleY = min(max(peakY + dy, 0), FootNet.size - 1)
          let value = exp(Double(map[sampleY * FootNet.size + sampleX] - peak))
          weight += value
          x += value * Double(sampleX)
          y += value * Double(sampleY)
        }
      }
      let scale = box.side / Double(FootNet.size)
      return [box.x + x / weight * scale, box.y + y / weight * scale, Double(peak)]
    }
  }
}
