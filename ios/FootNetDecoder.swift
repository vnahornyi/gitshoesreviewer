import Accelerate
import Foundation

// The crop's placement in the frame, used to map model coordinates back.
struct Crop {
  let x: Double
  let y: Double
  let side: Double
}

enum FootNetDecoder {
  static let size = 256
  static let joints = 8
  static let refineRadius = 2

  // Core ML returns float16 logits, matching the training decoder's input.
  static func decode(_ maps: UnsafePointer<Float>, crop box: Crop) -> [Double] {
    let plane = size * size
    return (0..<joints).flatMap { joint -> [Double] in
      let map = maps + joint * plane
      var peak: Float = 0
      var index: vDSP_Length = 0
      vDSP_maxvi(map, 1, &peak, &index, vDSP_Length(plane))
      let peakX = Int(index) % size
      let peakY = Int(index) / size

      var weight = 0.0
      var x = 0.0
      var y = 0.0
      for dy in -refineRadius...refineRadius {
        for dx in -refineRadius...refineRadius {
          let sampleX = min(max(peakX + dx, 0), size - 1)
          let sampleY = min(max(peakY + dy, 0), size - 1)
          let value = exp(Double(map[sampleY * size + sampleX] - peak))
          weight += value
          x += value * Double(sampleX)
          y += value * Double(sampleY)
        }
      }
      let scale = box.side / Double(size)
      let score = 1 / (1 + exp(-Double(peak)))
      return [box.x + x / weight * scale, box.y + y / weight * scale, score]
    }
  }
}

enum FootNetCoordinates {
  static func normalized(_ points: [Double], width: Double, height: Double) -> [Double] {
    stride(from: 0, to: points.count, by: 3).flatMap { point in
      [points[point] / width, points[point + 1] / height, points[point + 2]]
    }
  }
}
