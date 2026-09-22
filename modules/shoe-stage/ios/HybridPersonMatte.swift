import Accelerate
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
  // Rows of the copy Vision segments and of the stored matte; ShoeView scales it to the view.
  private static let inputHeight = 384
  private static let maxHeight = 256
  private let request: VNGeneratePersonSegmentationRequest = {
    let request = VNGeneratePersonSegmentationRequest()
    request.qualityLevel = .balanced
    request.outputPixelFormat = kCVPixelFormatType_OneComponent8
    return request
  }()
  private let queue = DispatchQueue(label: "shoe-stage.person-matte", qos: .userInitiated)
  private let lock = NSLock()
  private var isEnabled = false
  private var busy = false
  private var lastMs: Double = 0
  private var pool: CVPixelBufferPool?
  private var poolSize = (width: 0, height: 0)

  var enabled: Bool {
    get { lock.withLock { isEnabled } }
    set { lock.withLock { isEnabled = newValue } }
  }

  // Segmentation takes longer than a frame on older phones, so it runs beside the camera thread on a small copy and
  // frames that arrive while it is busy are skipped.
  func update(frame: any HybridFrameSpec) throws -> Double {
    let (enabled, busy, lastMs) = lock.withLock { (isEnabled, self.busy, self.lastMs) }
    guard enabled else { return 0 }
    guard !busy else { return lastMs }
    guard let frame = frame as? any NativeFrame,
          let sampleBuffer = frame.sampleBuffer,
          let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      throw PersonMatteError.unsupportedFrame
    }
    guard let copy = scaledCopy(of: pixelBuffer) else { return lastMs }
    lock.withLock { self.busy = true }
    queue.async { [weak self] in
      self?.segment(copy)
    }
    return lastMs
  }

  private func segment(_ image: CVPixelBuffer) {
    let started = CACurrentMediaTime()
    do {
      try VNImageRequestHandler(cvPixelBuffer: image, orientation: .up).perform([request])
      if let mask = request.results?.first?.pixelBuffer,
         let matte = Self.downsample(mask, frameWidth: CVPixelBufferGetWidth(image),
                                     frameHeight: CVPixelBufferGetHeight(image), time: started) {
        MatteStore.shared.publish(matte)
      }
    } catch {}
    let elapsed = (CACurrentMediaTime() - started) * 1000
    lock.withLock {
      lastMs = elapsed
      busy = false
    }
  }

  private func scaledCopy(of source: CVPixelBuffer) -> CVPixelBuffer? {
    guard CVPixelBufferGetPixelFormatType(source) == kCVPixelFormatType_32BGRA else { return nil }
    let sourceWidth = CVPixelBufferGetWidth(source)
    let sourceHeight = CVPixelBufferGetHeight(source)
    let height = min(Self.inputHeight, sourceHeight)
    let width = max(1, sourceWidth * height / sourceHeight)
    if pool == nil || poolSize != (width, height) {
      let attributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
      ]
      CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
      poolSize = (width, height)
    }
    var copy: CVPixelBuffer?
    guard let pool, CVPixelBufferPoolCreatePixelBuffer(nil, pool, &copy) == kCVReturnSuccess, let copy else { return nil }
    CVPixelBufferLockBaseAddress(source, .readOnly)
    CVPixelBufferLockBaseAddress(copy, [])
    defer {
      CVPixelBufferUnlockBaseAddress(copy, [])
      CVPixelBufferUnlockBaseAddress(source, .readOnly)
    }
    var from = vImage_Buffer(
      data: CVPixelBufferGetBaseAddress(source),
      height: vImagePixelCount(sourceHeight),
      width: vImagePixelCount(sourceWidth),
      rowBytes: CVPixelBufferGetBytesPerRow(source)
    )
    var to = vImage_Buffer(
      data: CVPixelBufferGetBaseAddress(copy),
      height: vImagePixelCount(height),
      width: vImagePixelCount(width),
      rowBytes: CVPixelBufferGetBytesPerRow(copy)
    )
    guard vImageScale_ARGB8888(&from, &to, nil, vImage_Flags(kvImageNoFlags)) == kvImageNoError else { return nil }
    return copy
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
