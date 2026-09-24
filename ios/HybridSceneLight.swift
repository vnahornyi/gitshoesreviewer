import AVFoundation
import Foundation
import NitroModules
import QuartzCore
import VisionCamera

// Mean sRGB-encoded colour (0…1) of the lower half of the upright frame.
struct SceneTone {
  let red: Float
  let green: Float
  let blue: Float
  let time: CFTimeInterval

  var luma: Float { 0.2126 * red + 0.7152 * green + 0.0722 * blue }
}

final class SceneToneStore {
  static let shared = SceneToneStore()
  private let lock = NSLock()
  private var tone: SceneTone?

  func publish(_ next: SceneTone) {
    lock.lock()
    tone = next
    lock.unlock()
  }

  func latest(maxAge: CFTimeInterval) -> SceneTone? {
    lock.lock()
    defer { lock.unlock() }
    guard let tone, CACurrentMediaTime() - tone.time <= maxAge else { return nil }
    return tone
  }
}

final class HybridSceneLight: HybridSceneLightSpec {
  // A sparse grid is enough for an average and costs microseconds.
  private static let grid = 32
  private let lock = NSLock()
  private var isEnabled = false

  var enabled: Bool {
    get { lock.withLock { isEnabled } }
    set { lock.withLock { isEnabled = newValue } }
  }

  func update(frame: any HybridFrameSpec) throws {
    guard enabled, let frame = frame as? any NativeFrame,
          let sampleBuffer = frame.sampleBuffer,
          let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
          CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else { return }
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let pixels = base.assumingMemoryBound(to: UInt8.self)
    var sum = SIMD3<Float>(0, 0, 0)
    for gy in 0..<Self.grid {
      let y = height / 2 + (gy * 2 + 1) * height / (4 * Self.grid)
      for gx in 0..<Self.grid {
        let x = (gx * 2 + 1) * width / (2 * Self.grid)
        let p = pixels + y * rowBytes + x * 4
        sum += SIMD3(Float(p[2]), Float(p[1]), Float(p[0]))
      }
    }
    let mean = sum / Float(Self.grid * Self.grid * 255)
    SceneToneStore.shared.publish(SceneTone(red: mean.x, green: mean.y, blue: mean.z, time: CACurrentMediaTime()))
  }
}
