import Accelerate
import AVFoundation
import Foundation
import NitroModules
import VisionCamera
import onnxruntime_objc

private enum Input {
  static let width = 192
  static let height = 256
  static let simccSplit = 2
  static let mean: [Float] = [123.675, 116.28, 103.53]
  static let std: [Float] = [58.395, 57.12, 57.375]
}

private struct ModelSpec {
  let resource: String
  let joints: Int
  // Indices in FOOT_JOINTS order: ankles, left big toe, small toe, heel, right big toe, small toe, heel.
  let footJoints: [Int]

  static func of(_ model: FootModel) -> ModelSpec {
    switch model {
    case .rtmwXL:
      return ModelSpec(resource: "rtmw-x-l-fp16", joints: 133, footJoints: [15, 16, 17, 18, 19, 20, 21, 22])
    case .rtmposeM:
      return ModelSpec(resource: "rtmpose-m-fp16", joints: 26, footJoints: [15, 16, 20, 22, 24, 21, 23, 25])
    }
  }
}

enum FootPoseError: Error, LocalizedError {
  case modelMissing(String)
  case notReady(String)
  case unsupportedFrame(String)

  var errorDescription: String? {
    switch self {
    case .modelMissing(let resource):
      return "\(resource) is not in the app bundle: put it in modules/foot-pose/model and run pod install"
    case .notReady(let status):
      return "detector is \(status)"
    case .unsupportedFrame(let reason):
      return "unsupported frame: \(reason)"
    }
  }
}

private struct Letterbox {
  let offsetX: Double
  let offsetY: Double
  let scaledWidth: Double
  let scaledHeight: Double
}

// Foot joints per side in FOOT_JOINTS order, and the minimum score for a point to shape the crop.
private enum Feet {
  static let joints: [(side: Int, indices: [Int])] = [(0, [2, 3, 4]), (1, [5, 6, 7])]
  static let ankles = [0, 1]
  static let minScore = 0.2
  // A foot FootNet is following needs this many points it is sure of, or it counts as lost. Three is what a crop can
  // be built from; on real frames the model is sure of 3.9 of its 8 points on average, so asking for more loses the
  // foot most of the time.
  static let trackedScore = 0.3
  static let trackedPoints = 3
  // A foot cannot change its apparent size by much in a thirtieth of a second, so the crop may only grow or shrink
  // by this much per frame. Without it a crop built from three points that happen to sit together collapses onto a
  // corner of the foot, the model sees even less in it, and it never recovers.
  static let sizeStep = 1.2
  // How long a crop keeps being looked at after the last frame the model was sure in it.
  static let lostAfter = FootMaskPreviewStore.retentionDuration
  // A walking foot is centimetres further on by the time the next frame arrives, so a crop placed where it was is a
  // crop it is walking out of. The model then grows unsure in it, the foot is dropped, and the whole-frame search
  // takes a quarter of a second to pick it up — which is what makes the shoe blink while walking. These lead the
  // crop by the speed of the last frames instead. The blend tames a noisy one-frame estimate, and the cap keeps a
  // bad one from flinging the crop off the foot entirely.
  static let motionBlend = 0.5
  static let maxLead = 0.5
  // How often the body model may search the whole frame for a foot FootNet is not following. Searching costs more
  // than a frame's whole budget, and a foot that is out of frame would otherwise have it searching on every one.
  static let searchInterval: CFTimeInterval = 0.25
}

private final class Runner {
  let spec: ModelSpec
  let session: ORTSession
  var footNet: FootNetRunner?
  private let inputData = NSMutableData(length: 3 * Input.height * Input.width * MemoryLayout<Float>.size)!
  private let simccXData: NSMutableData
  private let simccYData: NSMutableData
  private var letterbox = [UInt8](repeating: 0, count: Input.width * Input.height * 4)
  // Where to look for each foot next, in frame pixels, when it was last seen there, and the last search's points,
  // which seed a crop for a foot that has none.
  private var tracked: [Crop?] = [nil, nil]
  private var seenAt: [CFTimeInterval] = [0, 0]
  // How fast each foot's crop is moving across the frame, in pixels per second, to lead it on the next frame.
  private var motion: [(x: Double, y: Double)] = [(0, 0), (0, 0)]
  private var searched: [Double] = []
  private var searchedAt: CFTimeInterval = 0

  init(spec: ModelSpec, session: ORTSession) {
    self.spec = spec
    self.session = session
    simccXData = NSMutableData(length: spec.joints * Input.width * Input.simccSplit * MemoryLayout<Float>.size)!
    simccYData = NSMutableData(length: spec.joints * Input.height * Input.simccSplit * MemoryLayout<Float>.size)!
  }

  // The body model searches the whole frame; FootNet follows each foot in its own crop and says where it is next
  // frame. So the search only runs when a foot is missing, and the frames in between cost FootNet alone.
  func run(_ pixelBuffer: CVPixelBuffer, refine: Bool, maskPreview: Bool,
           maskThreshold: Double) throws -> FootPoseResult {
    let started = CACurrentMediaTime()
    let following = refine && footNet != nil
    let search = !following || (tracked.contains(where: { $0 == nil }) && started - searchedAt >= Feet.searchInterval)
    var points = [Double](repeating: 0, count: spec.footJoints.count * 3)
    var prepared = started
    var finished = started
    if search {
      let transform = try fillInput(from: pixelBuffer)
      prepared = CACurrentMediaTime()
      try session.run(
        withInputs: ["input": try tensor(inputData, shape: [1, 3, Input.height, Input.width])],
        outputs: [
          "simcc_x": try tensor(simccXData, shape: [1, spec.joints, Input.width * Input.simccSplit]),
          "simcc_y": try tensor(simccYData, shape: [1, spec.joints, Input.height * Input.simccSplit]),
        ],
        runOptions: nil
      )
      finished = CACurrentMediaTime()
      points = decode(transform)
      searched = points
      searchedAt = started
    }
    FootMaskPreviewStore.shared.beginFrame()
    let refinement = following
      ? self.refine(in: pixelBuffer, at: started, maskPreview: maskPreview, threshold: maskThreshold)
      : (points: [Double](repeating: 0, count: 2 * FootNet.joints * 3),
         maskPreviewMs: 0.0, crops: [Double](repeating: 0, count: 8))
    let width = Double(CVPixelBufferGetWidth(pixelBuffer))
    let height = Double(CVPixelBufferGetHeight(pixelBuffer))
    return FootPoseResult(
      points: points,
      refined: refinement.points,
      crops: refinement.crops,
      preprocessMs: (prepared - started) * 1000,
      inferenceMs: (finished - prepared) * 1000,
      refineMs: (CACurrentMediaTime() - finished) * 1000,
      maskPreviewMs: refinement.maskPreviewMs,
      maskPreviewStatus: footNet?.maskPreviewStatus ?? "loading"
    )
  }

  // Each foot's crop goes to FootNet, and its 8 points come back in frame coordinates. A crop the model was sure in
  // moves onto its own points for the next frame; one it was not sure in stays where it is for a moment, because the
  // foot is usually still there and the whole-frame search costs more than the frame's budget.
  private func refine(in pixelBuffer: CVPixelBuffer, at now: CFTimeInterval,
                      maskPreview: Bool, threshold: Double)
    -> (points: [Double], maskPreviewMs: Double, crops: [Double]) {
    let empty = [Double](repeating: 0, count: FootNet.joints * 3)
    guard let footNet else { return (empty + empty, 0, [Double](repeating: 0, count: 8)) }
    let width = Double(CVPixelBufferGetWidth(pixelBuffer))
    let height = Double(CVPixelBufferGetHeight(pixelBuffer))
    var refined = [[Double]](repeating: empty, count: Feet.joints.count)
    var sureCount = [Int](repeating: 0, count: Feet.joints.count)
    var maskPreviewMs = 0.0
    var crops = [Double](repeating: 0, count: Feet.joints.count * 4)
    for foot in Feet.joints {
      let following = tracked[foot.side]
      if following == nil { motion[foot.side] = (0, 0) }
      guard let observed = following ?? seed(for: foot, width: width, height: height) else {
        tracked[foot.side] = nil
        continue
      }
      // Where the foot will be, not where it was. A crop seeded by the whole-frame search has no history to lead it
      // with, so it is used as it is.
      let box = following == nil ? observed : lead(observed, side: foot.side, at: now)
      guard let output = try? footNet.run(pixelBuffer, crop: box, maskPreview: maskPreview,
                                          threshold: threshold, side: foot.side),
            output.points.count == FootNet.joints * 3 else {
        tracked[foot.side] = nil
        continue
      }
      let cropOffset = foot.side * 4
      crops[cropOffset] = box.x / width
      crops[cropOffset + 1] = box.y / height
      crops[cropOffset + 2] = box.side / width
      crops[cropOffset + 3] = footNet.brightness
      maskPreviewMs += output.maskPreviewMs
      // Every point the model found is passed on with its score, however low; whoever places the shoe decides what to
      // trust. Only whether to keep following this crop is decided here.
      let normalized = FootNetCoordinates.normalized(output.points, width: width, height: height)
      refined[foot.side] = normalized

      let sure = stride(from: 0, to: output.points.count, by: 3)
        .filter { output.points[$0 + 2] >= Feet.trackedScore }
        .map { (x: output.points[$0], y: output.points[$0 + 1]) }
      guard sure.count >= Feet.trackedPoints, let next = FootNetRunner.box(around: sure) else {
        // The crop that is kept is the one the foot was last seen in, never the led one: leading a led crop again
        // on every frame would walk it off the foot on its own.
        tracked[foot.side] = now - seenAt[foot.side] <= Feet.lostAfter ? observed : nil
        continue
      }
      // Follow the foot's new position, but hold the crop's size steady.
      let side = min(max(next.side, box.side / Feet.sizeStep), box.side * Feet.sizeStep)
      let elapsed = now - seenAt[foot.side]
      if following != nil, elapsed > 0, elapsed <= Feet.lostAfter {
        let moved = (x: (next.x + next.side / 2 - observed.x - observed.side / 2) / elapsed,
                     y: (next.y + next.side / 2 - observed.y - observed.side / 2) / elapsed)
        motion[foot.side] = (x: motion[foot.side].x + Feet.motionBlend * (moved.x - motion[foot.side].x),
                             y: motion[foot.side].y + Feet.motionBlend * (moved.y - motion[foot.side].y))
      }
      tracked[foot.side] = Crop(x: next.x + (next.side - side) / 2, y: next.y + (next.side - side) / 2, side: side)
      seenAt[foot.side] = now
      sureCount[foot.side] = sure.count
    }
    dropTheDouble(&refined, sureCount: sureCount, empty: empty)
    return (refined.flatMap { $0 }, maskPreviewMs, crops)
  }

  // Where to look for a foot that is moving: its last crop, carried on by the speed of the frames before it. The
  // offset is capped at half the crop, so a velocity read off one bad frame shifts the crop without losing the foot.
  private func lead(_ box: Crop, side: Int, at now: CFTimeInterval) -> Crop {
    let elapsed = now - seenAt[side]
    guard elapsed > 0, elapsed <= Feet.lostAfter else { return box }
    let limit = box.side * Feet.maxLead
    let dx = min(max(motion[side].x * elapsed, -limit), limit)
    let dy = min(max(motion[side].y * elapsed, -limit), limit)
    return Crop(x: box.x + dx, y: box.y + dy, side: box.side)
  }

  // Two crops can drift onto the same foot, and then both of them follow it for good. The less certain one is let go
  // and the body model picks its foot up again.
  private func dropTheDouble(_ refined: inout [[Double]], sureCount: [Int], empty: [Double]) {
    guard let left = tracked[0], let right = tracked[1] else { return }
    let apart = hypot(left.x + left.side / 2 - right.x - right.side / 2,
                      left.y + left.side / 2 - right.y - right.side / 2)
    guard apart < min(left.side, right.side) / 2 else { return }
    let loser = sureCount[0] < sureCount[1] ? 0 : 1
    tracked[loser] = nil
    refined[loser] = empty
  }

  // The crop to pick a foot up in, from what the body model last saw of it.
  private func seed(for foot: (side: Int, indices: [Int]), width: Double, height: Double) -> Crop? {
    let points = searched
    guard points.count == spec.footJoints.count * 3 else { return nil }
    var seen = foot.indices
      .filter { points[$0 * 3 + 2] >= Feet.minScore }
      .map { (x: points[$0 * 3] * width, y: points[$0 * 3 + 1] * height) }
    let ankle = Feet.ankles[foot.side]
    // Keep the ankle with the foot points: two visible toes alone make a toe-only seed when the heel is hidden.
    if points[ankle * 3 + 2] >= Feet.minScore {
      seen.append((x: points[ankle * 3] * width, y: points[ankle * 3 + 1] * height))
    }
    return seen.count >= 2 ? FootNetRunner.box(around: seen) : nil
  }

  private func tensor(_ data: NSMutableData, shape: [Int]) throws -> ORTValue {
    try ORTValue(tensorData: data, elementType: .float, shape: shape.map { NSNumber(value: $0) })
  }

  private func fillInput(from pixelBuffer: CVPixelBuffer) throws -> Letterbox {
    guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else {
      throw FootPoseError.unsupportedFrame("expected BGRA, use pixelFormat 'rgb'")
    }
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let scale = max(Double(width) / Double(Input.width), Double(height) / Double(Input.height))
    let scaledWidth = min(Input.width, Int((Double(width) / scale).rounded()))
    let scaledHeight = min(Input.height, Int((Double(height) / scale).rounded()))
    let offsetX = (Input.width - scaledWidth) / 2
    let offsetY = (Input.height - scaledHeight) / 2
    let rowBytes = Input.width * 4

    var source = vImage_Buffer(
      data: CVPixelBufferGetBaseAddress(pixelBuffer),
      height: vImagePixelCount(height),
      width: vImagePixelCount(width),
      rowBytes: CVPixelBufferGetBytesPerRow(pixelBuffer)
    )
    let scaled = letterbox.withUnsafeMutableBytes { bytes -> vImage_Error in
      bytes.initializeMemory(as: UInt8.self, repeating: 0)
      var destination = vImage_Buffer(
        data: bytes.baseAddress! + offsetY * rowBytes + offsetX * 4,
        height: vImagePixelCount(scaledHeight),
        width: vImagePixelCount(scaledWidth),
        rowBytes: rowBytes
      )
      return vImageScale_ARGB8888(&source, &destination, nil, vImage_Flags(kvImageNoFlags))
    }
    guard scaled == kvImageNoError else { throw FootPoseError.unsupportedFrame("vImage scale failed (\(scaled))") }

    let plane = Input.width * Input.height
    let input = inputData.mutableBytes.assumingMemoryBound(to: Float.self)
    letterbox.withUnsafeBufferPointer { pixels in
      for index in 0..<plane {
        let blue = Float(pixels[index * 4])
        let green = Float(pixels[index * 4 + 1])
        let red = Float(pixels[index * 4 + 2])
        input[index] = (red - Input.mean[0]) / Input.std[0]
        input[plane + index] = (green - Input.mean[1]) / Input.std[1]
        input[2 * plane + index] = (blue - Input.mean[2]) / Input.std[2]
      }
    }
    return Letterbox(
      offsetX: Double(offsetX),
      offsetY: Double(offsetY),
      scaledWidth: Double(scaledWidth),
      scaledHeight: Double(scaledHeight)
    )
  }

  private func decode(_ letterbox: Letterbox) -> [Double] {
    let binsX = Input.width * Input.simccSplit
    let binsY = Input.height * Input.simccSplit
    let simccX = simccXData.bytes.assumingMemoryBound(to: Float.self)
    let simccY = simccYData.bytes.assumingMemoryBound(to: Float.self)

    return spec.footJoints.flatMap { joint -> [Double] in
      let (binX, peakX) = argmax(simccX + joint * binsX, count: binsX)
      let (binY, peakY) = argmax(simccY + joint * binsY, count: binsY)
      let x = (Double(binX) / Double(Input.simccSplit) - letterbox.offsetX) / letterbox.scaledWidth
      let y = (Double(binY) / Double(Input.simccSplit) - letterbox.offsetY) / letterbox.scaledHeight
      return [x, y, Double(min(peakX, peakY))]
    }
  }

  private func argmax(_ values: UnsafePointer<Float>, count: Int) -> (Int, Float) {
    var peak: Float = 0
    var index: vDSP_Length = 0
    vDSP_maxvi(values, 1, &peak, &index, vDSP_Length(count))
    return (Int(index), peak)
  }
}

class HybridFootPoseDetector: HybridFootPoseDetectorSpec {
  private let lock = NSLock()
  private let loading = DispatchQueue(label: "foot-pose.load", qos: .userInitiated)
  private var runner: Runner?
  private var requested: FootModel?
  private var loadStatus = "idle"
  private var refineFeet = false
  private var maskPreviewEnabled = false
  private var maskThresholdValue = 0.5

  var refine: Bool {
    get { lock.withLock { refineFeet } }
    set { lock.withLock { refineFeet = newValue } }
  }

  var maskPreview: Bool {
    get { lock.withLock { maskPreviewEnabled } }
    set {
      lock.withLock { maskPreviewEnabled = newValue }
      if !newValue { FootMaskPreviewStore.shared.clearAll() }
    }
  }

  var maskThreshold: Double {
    get { lock.withLock { maskThresholdValue } }
    set {
      let value = min(max(newValue, 0.0), 1.0)
      let changed = lock.withLock {
        guard maskThresholdValue != value else { return false }
        maskThresholdValue = value
        return true
      }
      if changed { FootMaskPreviewStore.shared.clearAll() }
    }
  }

  var status: String {
    lock.lock()
    defer { lock.unlock() }
    return loadStatus
  }

  func load(model: FootModel) throws {
    lock.lock()
    guard requested != model else {
      lock.unlock()
      return
    }
    requested = model
    runner = nil
    loadStatus = "loading \(model.stringValue)"
    lock.unlock()

    loading.async { [weak self] in
      self?.build(model)
    }
  }

  func detect(frame: any HybridFrameSpec) throws -> FootPoseResult {
    lock.lock()
    let runner = self.runner
    let status = loadStatus
    let refine = refineFeet
    let maskPreview = maskPreviewEnabled
    let maskThreshold = maskThresholdValue
    lock.unlock()
    guard let runner else { throw FootPoseError.notReady(status) }
    guard let frame = frame as? any NativeFrame,
          let sampleBuffer = frame.sampleBuffer,
          let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      throw FootPoseError.unsupportedFrame("no pixel buffer")
    }
    return try runner.run(pixelBuffer, refine: refine, maskPreview: maskPreview,
                          maskThreshold: maskThreshold)
  }

  private func build(_ model: FootModel) {
    let spec = ModelSpec.of(model)
    let outcome: Result<Runner, Error> = Result {
      guard let path = Bundle.main.path(forResource: spec.resource, ofType: "onnx")
        ?? Bundle(for: HybridFootPoseDetector.self).path(forResource: spec.resource, ofType: "onnx") else {
        throw FootPoseError.modelMissing(spec.resource)
      }
      let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("foot-pose-coreml/\(spec.resource)", isDirectory: true)
      try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)

      let options = try ORTSessionOptions()
      try options.appendCoreMLExecutionProvider(withOptionsV2: [
        "ModelFormat": "MLProgram",
        "MLComputeUnits": "ALL",
        "ModelCacheDirectory": cache.path,
      ])
      let environment = try ORTEnv(loggingLevel: .warning)
      let session = try ORTSession(env: environment, modelPath: path, sessionOptions: options)
      let built = Runner(spec: spec, session: session)
      built.footNet = try FootNetRunner.load(from: Bundle(for: HybridFootPoseDetector.self))
      return built
    }

    lock.lock()
    defer { lock.unlock() }
    guard requested == model else { return }
    switch outcome {
    case .success(let built):
      runner = built
      loadStatus = "ready"
    case .failure(let error):
      loadStatus = "error: \(error.localizedDescription)"
    }
  }
}
