import CoreGraphics
import Foundation
import UIKit

struct FootMaskPreviewFrame {
  let pixels: Data
  let x: Double
  let y: Double
  let width: Double
  let height: Double
}

final class FootMaskPreviewStore {
  static let shared = FootMaskPreviewStore()

  private let lock = NSLock()
  private var frames: [FootMaskPreviewFrame?] = [nil, nil]

  func beginFrame() {
    lock.withLock { frames = [nil, nil] }
  }

  func set(_ frame: FootMaskPreviewFrame, side: Int) {
    lock.withLock {
      guard frames.indices.contains(side) else { return }
      frames[side] = frame
    }
  }

  func snapshot() -> [FootMaskPreviewFrame?] {
    lock.withLock { frames }
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
  }

  func clear() {
    masks.forEach { $0.contents = nil }
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
