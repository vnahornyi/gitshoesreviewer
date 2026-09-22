import AVFoundation
import Foundation
import NitroModules
import QuartzCore
import Vision
import VisionCamera

// A person mask aligned with the upright camera frame: 0 background, 255 person.
struct Matte {
  let width: Int
  let height: Int
  let bytes: [UInt8]
  let time: CFTimeInterval
}

// The frame processor writes the latest matte, ShoeView reads it on the main thread.
final class MatteStore {
  static let shared = MatteStore()
  private let lock = NSLock()
  private var matte: Matte?

  func publish(_ next: Matte) {
    lock.lock()
    matte = next
    lock.unlock()
  }

  func latest(maxAge: CFTimeInterval) -> Matte? {
    lock.lock()
    defer { lock.unlock() }
    guard let matte, CACurrentMediaTime() - matte.time <= maxAge else { return nil }
    return matte
  }
}

enum PersonMatteError: Error, LocalizedError {
  case unsupportedFrame

  var errorDescription: String? { "unsupported frame: no pixel buffer" }
}

final class HybridPersonMatte: HybridPersonMatteSpec {
  // Rows of the stored matte; ShoeView scales it to the view, so more would only cost time.
  private static let maxHeight = 256
  private let request: VNGeneratePersonSegmentationRequest = {
    let request = VNGeneratePersonSegmentationRequest()
    request.qualityLevel = .balanced
    request.outputPixelFormat = kCVPixelFormatType_OneComponent8
    return request
  }()

  func update(frame: any HybridFrameSpec) throws -> Double {
    guard let frame = frame as? any NativeFrame,
          let sampleBuffer = frame.sampleBuffer,
          let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      throw PersonMatteError.unsupportedFrame
    }
    let started = CACurrentMediaTime()
    try VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform([request])
    if let mask = request.results?.first?.pixelBuffer,
       let matte = Self.downsample(mask, frameWidth: CVPixelBufferGetWidth(pixelBuffer),
                                   frameHeight: CVPixelBufferGetHeight(pixelBuffer), time: started) {
      MatteStore.shared.publish(matte)
    }
    return (CACurrentMediaTime() - started) * 1000
  }

  // Nearest-neighbour copy into a buffer with the frame's aspect ratio (Vision's mask may use another one).
  private static func downsample(_ mask: CVPixelBuffer, frameWidth: Int, frameHeight: Int, time: CFTimeInterval) -> Matte? {
    CVPixelBufferLockBaseAddress(mask, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(mask), frameWidth > 0, frameHeight > 0 else { return nil }
    let sourceWidth = CVPixelBufferGetWidth(mask)
    let sourceHeight = CVPixelBufferGetHeight(mask)
    let rowBytes = CVPixelBufferGetBytesPerRow(mask)
    let height = min(maxHeight, sourceHeight)
    let width = max(1, Int((Double(height) * Double(frameWidth) / Double(frameHeight)).rounded()))
    let source = base.assumingMemoryBound(to: UInt8.self)
    var bytes = [UInt8](repeating: 0, count: width * height)
    for y in 0..<height {
      let row = source + (y * sourceHeight / height) * rowBytes
      for x in 0..<width {
        bytes[y * width + x] = row[x * sourceWidth / width]
      }
    }
    return Matte(width: width, height: height, bytes: bytes, time: time)
  }
}
