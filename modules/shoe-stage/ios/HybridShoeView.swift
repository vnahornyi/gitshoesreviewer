import CoreGraphics
import CoreImage
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

// Estimates, to be tuned on device: a frame averaging this luma gets the lights at their base intensity.
private enum Look {
  static let maxAge: CFTimeInterval = 0.5
  static let keyIntensity: Float = 2500
  static let fillIntensity: Float = 800
  static let referenceLuma: Float = 0.45
  static let exposureRange: ClosedRange<Float> = 0.35...1.8
  // How much of the frame's colour cast the lights take on (0 keeps them white).
  static let tintStrength: Float = 0.6
  // Per-frame smoothing of the light, so it follows exposure changes without flicker.
  static let smoothing: Float = 0.15
  static let blurRadius: Double = 0.7
  static let grainAtReference: Float = 0.025
  static let grainRange: ClosedRange<Float> = 0.01...0.07
}

// Written on the main thread, read by RealityKit's post-process callback on the render thread.
private final class CameraLook {
  private let lock = NSLock()
  private var grain: Float?

  func set(grain next: Float?) {
    lock.lock()
    grain = next
    lock.unlock()
  }

  func current() -> Float? {
    lock.lock()
    defer { lock.unlock() }
    return grain
  }
}

final class HybridShoeView: HybridShoeViewSpec {
  private let arView = ARView(frame: .zero, cameraMode: .nonAR, automaticallyConfigureSession: false)
  private let camera = PerspectiveCamera()
  private let content = AnchorEntity(world: .zero)
  private var templates: [ShoeSide: Entity] = [:]
  private var placed: [Int: (side: ShoeSide, entity: Entity)] = [:]
  private let matteLayer = CALayer()
  private let key = DirectionalLight()
  private let fill = DirectionalLight()
  private var tone = SIMD4<Float>(1, 1, 1, 1)
  private let look = CameraLook()
  private var ciContext: CIContext?

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
  var matchCamera: Bool = false

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

    key.light.intensity = Look.keyIntensity
    key.look(at: [0, -0.5, -1.5], from: [0.5, 1.5, 0.5], relativeTo: nil)
    content.addChild(key)

    fill.light.intensity = Look.fillIntensity
    fill.look(at: [0, -0.5, -1.5], from: [-1, 0.2, 0.5], relativeTo: nil)
    content.addChild(fill)

    arView.renderCallbacks.postProcess = { [weak self] context in
      self?.postProcess(context)
    }

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
    applyLook()
  }

  // Scales and tints the lights toward the frame's average exposure and colour cast, and sets the grain for dim scenes.
  private func applyLook() {
    guard matchCamera, let scene = SceneToneStore.shared.latest(maxAge: Look.maxAge) else {
      tone = [1, 1, 1, 1]
      setLights(tone)
      look.set(grain: nil)
      return
    }
    let luma = max(scene.luma, 0.01)
    let exposure = min(max(luma / Look.referenceLuma, Look.exposureRange.lowerBound), Look.exposureRange.upperBound)
    let cast = SIMD3(scene.red, scene.green, scene.blue) / luma
    let tint = SIMD3<Float>(1, 1, 1) + Look.tintStrength * (cast - SIMD3<Float>(1, 1, 1))
    let target = SIMD4(tint.x, tint.y, tint.z, exposure)
    tone += Look.smoothing * (target - tone)
    setLights(tone)
    let grain = Look.grainAtReference * Look.referenceLuma / luma
    look.set(grain: min(max(grain, Look.grainRange.lowerBound), Look.grainRange.upperBound))
  }

  private func setLights(_ tone: SIMD4<Float>) {
    let largest = max(tone.x, tone.y, tone.z, 0.001)
    let color = UIColor(
      red: CGFloat(tone.x / largest),
      green: CGFloat(tone.y / largest),
      blue: CGFloat(tone.z / largest),
      alpha: 1
    )
    key.light.color = color
    fill.light.color = color
    key.light.intensity = Look.keyIntensity * tone.w
    fill.light.intensity = Look.fillIntensity * tone.w
  }

  // A camera never shows a perfectly sharp, noiseless edge: soften the render a little and add moving grain on the shoe only.
  private func postProcess(_ context: ARView.PostProcessContext) {
    let passThrough = {
      guard let blit = context.commandBuffer.makeBlitCommandEncoder() else { return }
      blit.copy(from: context.sourceColorTexture, to: context.targetColorTexture)
      blit.endEncoding()
    }
    guard let grain = look.current(), let source = CIImage(mtlTexture: context.sourceColorTexture) else {
      passThrough()
      return
    }
    if ciContext == nil {
      ciContext = CIContext(mtlDevice: context.device, options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
    }
    let extent = source.extent
    let blurred = source.clampedToExtent().applyingGaussianBlur(sigma: Look.blurRadius).cropped(to: extent)
    let shift = CGAffineTransform(translationX: CGFloat.random(in: 0...512), y: CGFloat.random(in: 0...512))
    let noise = CIFilter(name: "CIRandomGenerator")!.outputImage!
      .transformed(by: shift)
      .applyingFilter("CIColorMatrix", parameters: [
        "inputRVector": CIVector(x: CGFloat(grain), y: 0, z: 0, w: 0),
        "inputGVector": CIVector(x: CGFloat(grain), y: 0, z: 0, w: 0),
        "inputBVector": CIVector(x: CGFloat(grain), y: 0, z: 0, w: 0),
        "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 0),
        "inputBiasVector": CIVector(x: -CGFloat(grain) / 2, y: -CGFloat(grain) / 2, z: -CGFloat(grain) / 2, w: 0),
      ])
      .cropped(to: extent)
    // Keep the grain inside the shoe's alpha so the transparent background stays clean.
    let masked = noise.applyingFilter("CIBlendWithAlphaMask", parameters: [
      kCIInputBackgroundImageKey: CIImage.empty().cropped(to: extent),
      kCIInputMaskImageKey: blurred,
    ])
    let output = masked.applyingFilter("CIAdditionCompositing", parameters: [kCIInputBackgroundImageKey: blurred])
    let destination = CIRenderDestination(mtlTexture: context.targetColorTexture, commandBuffer: context.commandBuffer)
    destination.isFlipped = false
    do {
      _ = try ciContext?.startTask(toRender: output, to: destination)
    } catch {
      passThrough()
    }
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
