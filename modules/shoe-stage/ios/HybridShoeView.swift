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
}

final class HybridShoeView: HybridShoeViewSpec {
  private let arView = ARView(frame: .zero, cameraMode: .nonAR, automaticallyConfigureSession: false)
  private let camera = PerspectiveCamera()
  private let content = AnchorEntity(world: .zero)
  private var templates: [ShoeSide: Entity] = [:]
  private var placed: [Int: (side: ShoeSide, entity: Entity)] = [:]

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
