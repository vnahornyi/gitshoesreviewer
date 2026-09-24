import Foundation

@main
struct CoordinateProbe {
  static func main() {
    let size = FootNetDecoder.size
    var maps = [Float](repeating: -3, count: FootNetDecoder.joints * size * size)
    let expectedCropPoints = (0..<FootNetDecoder.joints).map { joint in
      (x: 40 + joint * 20, y: 55 + joint * 17)
    }
    for (joint, point) in expectedCropPoints.enumerated() {
      maps[joint * size * size + point.y * size + point.x] = 1
    }

    let crop = Crop(x: 120.25, y: 200.5, side: 448)
    let frameWidth = 1472.0
    let frameHeight = 828.0
    let framePoints = maps.withUnsafeBufferPointer {
      FootNetDecoder.decode($0.baseAddress!, crop: crop)
    }
    let normalized = FootNetCoordinates.normalized(framePoints, width: frameWidth, height: frameHeight)

    for (joint, cropPoint) in expectedCropPoints.enumerated() {
      let expectedFrameX = crop.x + Double(cropPoint.x) * crop.side / Double(size)
      let expectedFrameY = crop.y + Double(cropPoint.y) * crop.side / Double(size)
      precondition(abs(framePoints[joint * 3] - expectedFrameX) < 1e-9)
      precondition(abs(framePoints[joint * 3 + 1] - expectedFrameY) < 1e-9)
      precondition(abs(framePoints[joint * 3 + 2] - 1 / (1 + exp(-1))) < 1e-12)
      precondition(abs(normalized[joint * 3] - expectedFrameX / frameWidth) < 1e-12)
      precondition(abs(normalized[joint * 3 + 1] - expectedFrameY / frameHeight) < 1e-12)
    }
    print("crop → frame pixels → normalized frame coordinates: PASS (landscape 1472×828)")
  }
}
