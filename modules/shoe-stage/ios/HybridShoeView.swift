import CoreGraphics
import Foundation
import NitroModules
import RealityKit
import UIKit

// In the normalized shoe (heel at the origin, sole on y = 0, toe at z = 1).
private enum Fit {
  static let shinRadius: Float = 0.1
  static let shinBottom: Float = 0.28
  static let shinTop: Float = 2.0
  static let shinForward: Float = 0.2
  static let shadowWidth: Float = 0.6
  static let shadowLength: Float = 1.3
  static let shadowOpacity: Float = 0.55
  // The leg strip above the collar, where the person matte replaces the shoe with the real leg.
  static let legHalfWidth: Float = 0.35
  static let legProbe: Float = 0.6
  static let collarMargin: Float = 0.03
}

private enum Matting {
  // Older than this, the matte no longer matches the frame on screen: fall back to the shin cylinder.
  static let maxAge: CFTimeInterval = 0.3
  static let personThreshold: UInt8 = 128
  static let shinName = "shin-occluder"
}

final class HybridShoeView: HybridShoeViewSpec {
  private let arView = ARView(frame: .zero, cameraMode: .nonAR, automaticallyConfigureSession: false)
  private let camera = PerspectiveCamera()
  private let content = AnchorEntity(world: .zero)
  private var templates: [ShoeSide: Entity] = [:]
  private var placed: [Int: (side: ShoeSide, entity: Entity)] = [:]
  private let matteLayer = CALayer()

  var view: UIView { arView }

  var model: String = "" {
    didSet {
      guard model != oldValue else { return }
      let name = model
      DispatchQueue.main.async { self.load(name) }
    }
  }

  var verticalFovDegrees: Double = 60 {
    didSet {
      let degrees = Float(verticalFovDegrees)
      DispatchQueue.main.async { self.camera.camera.fieldOfViewInDegrees = degrees }
    }
  }

  var legMatte: Bool = false

  var shoes: [ShoePose] = [] {
    didSet {
      let poses = shoes
      DispatchQueue.main.async { self.place(poses) }
    }
  }

  override init() {
    super.init()
    arView.environment.background = .color(.clear)
    arView.backgroundColor = .clear
    arView.isOpaque = false
    arView.isUserInteractionEnabled = false
    arView.renderOptions.insert(.disableMotionBlur)

    camera.camera.near = 0.01
    camera.camera.far = 20
    // The app passes the frame's vertical field of view; make RealityKit read it that way.
    if #available(iOS 18.0, *) {
      camera.camera.fieldOfViewOrientation = .vertical
    }
    content.addChild(camera)

    let key = DirectionalLight()
    key.light.intensity = 2500
    key.look(at: [0, -0.5, -1.5], from: [0.5, 1.5, 0.5], relativeTo: nil)
    content.addChild(key)

    let fill = DirectionalLight()
    fill.light.intensity = 800
    fill.look(at: [0, -0.5, -1.5], from: [-1, 0.2, 0.5], relativeTo: nil)
    content.addChild(fill)

    arView.scene.addAnchor(content)
  }

  private func load(_ name: String) {
    for (_, shoe) in placed { shoe.entity.removeFromParent() }
    placed = [:]
    templates = [:]
    for side in [ShoeSide.left, .right] {
      guard let url = Bundle.main.url(forResource: "\(name)-\(side.stringValue)", withExtension: "usdz"),
            let shoe = try? Entity.load(contentsOf: url) else { continue }
      let entity = Entity()
      entity.addChild(shoe)
      entity.addChild(shinOccluder())
      if let shadow = contactShadow() {
        entity.addChild(shadow)
      }
      templates[side] = entity
    }
  }

  private func place(_ poses: [ShoePose]) {
    let wanted = Set(poses.map { Int($0.id) })
    for (id, shoe) in placed where !wanted.contains(id) {
      shoe.entity.removeFromParent()
      placed[id] = nil
    }
    for pose in poses {
      let id = Int(pose.id)
      guard pose.transform.count == 16, let template = templates[pose.side] else { continue }
      let entity: Entity
      if let existing = placed[id], existing.side == pose.side {
        entity = existing.entity
      } else {
        placed[id]?.entity.removeFromParent()
        entity = template.clone(recursive: true)
        content.addChild(entity)
        placed[id] = (pose.side, entity)
      }
      entity.transform = Transform(matrix: matrix(pose.transform))
    }
    applyMatte(poses)
  }

  // Makes the shoe transparent where the camera shows the person above the shoe's collar, so the real leg comes out of it.
  private func applyMatte(_ poses: [ShoePose]) {
    let matte = legMatte ? MatteStore.shared.latest(maxAge: Matting.maxAge) : nil
    for (_, shoe) in placed {
      shoe.entity.findEntity(named: Matting.shinName)?.isEnabled = matte == nil
    }
    guard let matte, arView.bounds.width > 0 else {
      arView.layer.mask = nil
      return
    }
    let strips = poses.compactMap { legStrip($0, matte: matte) }
    var rgba = [UInt8](repeating: 255, count: matte.width * matte.height * 4)
    for y in 0..<matte.height {
      for x in 0..<matte.width {
        let index = y * matte.width + x
        guard matte.bytes[index] >= Matting.personThreshold else { continue }
        let point = SIMD2<Float>(Float(x) + 0.5, Float(y) + 0.5)
        if strips.contains(where: { $0.contains(point) }) {
          rgba[index * 4 + 3] = 0
          rgba[index * 4] = 0
          rgba[index * 4 + 1] = 0
          rgba[index * 4 + 2] = 0
        }
      }
    }
    guard let provider = CGDataProvider(data: Data(rgba) as CFData),
          let image = CGImage(
            width: matte.width,
            height: matte.height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: matte.width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
          ) else { return }
    CATransaction.begin()
    CATransaction.setDisableActions(true)
    matteLayer.frame = arView.bounds
    matteLayer.contents = image
    arView.layer.mask = matteLayer
    CATransaction.commit()
  }

  // The part of the image above one shoe's collar and around its shin, in matte pixels.
  private func legStrip(_ pose: ShoePose, matte: Matte) -> LegStrip? {
    guard pose.transform.count == 16 else { return nil }
    let model = matrix(pose.transform)
    let focal = Float(matte.height) / 2 / tan(Float(verticalFovDegrees) * .pi / 360)
    func project(_ local: SIMD3<Float>) -> SIMD2<Float>? {
      let p = model * SIMD4(local, 1)
      guard p.z < -0.01 else { return nil }
      return SIMD2(Float(matte.width) / 2 + focal * p.x / -p.z, Float(matte.height) / 2 - focal * p.y / -p.z)
    }
    guard let collar = project([0, Fit.shinBottom, Fit.shinForward]),
          let above = project([0, Fit.shinBottom + Fit.legProbe, Fit.shinForward]) else { return nil }
    let rise = above - collar
    let risePixels = simd_length(rise)
    guard risePixels > 1 else { return nil }
    let shoePixels = risePixels / Fit.legProbe
    return LegStrip(
      origin: collar,
      up: rise / risePixels,
      halfWidth: Fit.legHalfWidth * shoePixels,
      margin: Fit.collarMargin * shoePixels
    )
  }

  // Hides the parts of the shoe behind the real shin, so the leg looks like it goes into the shoe.
  private func shinOccluder() -> ModelEntity {
    let height = Fit.shinTop - Fit.shinBottom
    let mesh = MeshResource.generateBox(
      width: 2 * Fit.shinRadius,
      height: height,
      depth: 2 * Fit.shinRadius,
      cornerRadius: Fit.shinRadius * 0.99
    )
    let occluder = ModelEntity(mesh: mesh, materials: [OcclusionMaterial()])
    occluder.name = Matting.shinName
    occluder.position = [0, Fit.shinBottom + height / 2, Fit.shinForward]
    return occluder
  }

  private func contactShadow() -> ModelEntity? {
    guard let image = radialFalloff(size: 128),
          let texture = try? TextureResource.generate(from: image, options: .init(semantic: .raw)) else { return nil }
    var material = UnlitMaterial(color: .black)
    material.blending = .transparent(opacity: .init(scale: Fit.shadowOpacity, texture: .init(texture)))
    let plane = ModelEntity(
      mesh: .generatePlane(width: Fit.shadowWidth, depth: Fit.shadowLength),
      materials: [material]
    )
    plane.position = [0, 0.002, 0.5]
    return plane
  }

  private func radialFalloff(size: Int) -> CGImage? {
    guard let context = CGContext(
      data: nil,
      width: size,
      height: size,
      bitsPerComponent: 8,
      bytesPerRow: size,
      space: CGColorSpaceCreateDeviceGray(),
      bitmapInfo: CGImageAlphaInfo.none.rawValue
    ), let gradient = CGGradient(
      colorsSpace: CGColorSpaceCreateDeviceGray(),
      colors: [CGColor(gray: 1, alpha: 1), CGColor(gray: 0.35, alpha: 1), CGColor(gray: 0, alpha: 1)] as CFArray,
      locations: [0, 0.55, 1]
    ) else { return nil }
    let center = CGPoint(x: size / 2, y: size / 2)
    context.drawRadialGradient(
      gradient,
      startCenter: center,
      startRadius: 0,
      endCenter: center,
      endRadius: CGFloat(size) / 2,
      options: []
    )
    return context.makeImage()
  }

  private func matrix(_ values: [Double]) -> simd_float4x4 {
    let v = values.map { Float($0) }
    return simd_float4x4(
      SIMD4(v[0], v[1], v[2], v[3]),
      SIMD4(v[4], v[5], v[6], v[7]),
      SIMD4(v[8], v[9], v[10], v[11]),
      SIMD4(v[12], v[13], v[14], v[15])
    )
  }
}

private struct LegStrip {
  let origin: SIMD2<Float>
  let up: SIMD2<Float>
  let halfWidth: Float
  let margin: Float

  func contains(_ point: SIMD2<Float>) -> Bool {
    let offset = point - origin
    let along = simd_dot(offset, up)
    let across = abs(offset.x * up.y - offset.y * up.x)
    return along > -margin && across < halfWidth
  }
}
