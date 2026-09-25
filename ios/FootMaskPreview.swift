import CoreGraphics
import Foundation
import QuartzCore
import UIKit

struct FootMaskPreviewFrame {
  let pixels: Data
  let x: Double
  let y: Double
  let width: Double
  let height: Double
  let hasForeground: Bool
}

final class FootMaskPreviewStore {
  static let shared = FootMaskPreviewStore()
  // Keep a usable preview only as long as the tracker itself tolerates a weak observation.
  static let retentionDuration: CFTimeInterval = 0.3

  private struct StoredFrame {
    let frame: FootMaskPreviewFrame
    let updatedAt: CFTimeInterval
  }

  private let lock = NSLock()
  private var frames: [StoredFrame?] = [nil, nil]

  func beginFrame() {
    lock.withLock { discardExpired(at: CACurrentMediaTime()) }
  }

  func set(_ frame: FootMaskPreviewFrame, side: Int) {
    lock.withLock {
      guard frames.indices.contains(side) else { return }
      guard frame.hasForeground else {
        guard let previous = frames[side], Self.isWithinOneMaskPixel(previous.frame, frame) else {
          frames[side] = nil
          return
        }
        return
      }
      frames[side] = StoredFrame(frame: frame, updatedAt: CACurrentMediaTime())
    }
  }

  func snapshot() -> [FootMaskPreviewFrame?] {
    lock.withLock {
      discardExpired(at: CACurrentMediaTime())
      return frames.map { $0?.frame }
    }
  }

  func clearAll() {
    lock.withLock { frames = [nil, nil] }
  }

  private func discardExpired(at now: CFTimeInterval) {
    for index in frames.indices {
      guard let frame = frames[index], now - frame.updatedAt > Self.retentionDuration else { continue }
      frames[index] = nil
    }
  }

  private static func isWithinOneMaskPixel(_ previous: FootMaskPreviewFrame,
                                           _ current: FootMaskPreviewFrame) -> Bool {
    let xTolerance = max(previous.width, current.width) / Double(FootNet.maskPreviewSize)
    let yTolerance = max(previous.height, current.height) / Double(FootNet.maskPreviewSize)
    return abs(previous.x - current.x) <= xTolerance
      && abs(previous.y - current.y) <= yTolerance
      && abs(previous.width - current.width) <= xTolerance
      && abs(previous.height - current.height) <= yTolerance
  }
}

final class FootMaskOverlayView: UIView {
  private let masks = [CALayer(), CALayer()]

  override init(frame: CGRect) {
    super.init(frame: frame)
    isUserInteractionEnabled = false
    isOpaque = false
    clipsToBounds = true
    for mask in masks {
      mask.contentsGravity = .resize
      mask.magnificationFilter = .linear
      layer.addSublayer(mask)
    }
  }

  required init?(coder: NSCoder) {
    fatalError("init(coder:) has not been implemented")
  }

  override func layoutSubviews() {
    super.layoutSubviews()
    refresh()
  }

  func refresh() {
    let frames = FootMaskPreviewStore.shared.snapshot()
    CATransaction.begin()
    CATransaction.setDisableActions(true)
    for index in masks.indices {
      guard let frame = frames[index], let image = makeImage(frame.pixels) else {
        masks[index].contents = nil
        continue
      }
      masks[index].contents = image
      masks[index].frame = CGRect(
        x: frame.x * bounds.width,
        y: frame.y * bounds.height,
        width: frame.width * bounds.width,
        height: frame.height * bounds.height
      )
    }
    CATransaction.commit()
  }

  func clear() {
    CATransaction.begin()
    CATransaction.setDisableActions(true)
    masks.forEach { $0.contents = nil }
    CATransaction.commit()
  }

  private func makeImage(_ pixels: Data) -> CGImage? {
    let size = 64
    guard pixels.count == size * size * 4,
          let provider = CGDataProvider(data: pixels as CFData) else { return nil }
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
                                  | CGBitmapInfo.byteOrder32Big.rawValue)
    return CGImage(
      width: size,
      height: size,
      bitsPerComponent: 8,
      bitsPerPixel: 32,
      bytesPerRow: size * 4,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: bitmapInfo,
      provider: provider,
      decode: nil,
      shouldInterpolate: true,
      intent: .defaultIntent
    )
  }
}
