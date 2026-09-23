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
}

private final class Runner {
  let spec: ModelSpec
  let session: ORTSession
  var footNet: FootNetRunner?
  private let inputData = NSMutableData(length: 3 * Input.height * Input.width * MemoryLayout<Float>.size)!
  private let simccXData: NSMutableData
  private let simccYData: NSMutableData
  private var letterbox = [UInt8](repeating: 0, count: Input.width * Input.height * 4)

  init(spec: ModelSpec, session: ORTSession) {
    self.spec = spec
    self.session = session
    simccXData = NSMutableData(length: spec.joints * Input.width * Input.simccSplit * MemoryLayout<Float>.size)!
    simccYData = NSMutableData(length: spec.joints * Input.height * Input.simccSplit * MemoryLayout<Float>.size)!
  }

  func run(_ pixelBuffer: CVPixelBuffer, refine: Bool) throws -> FootPoseResult {
    let started = CACurrentMediaTime()
    let transform = try fillInput(from: pixelBuffer)
    let prepared = CACurrentMediaTime()
    try session.run(
      withInputs: ["input": try tensor(inputData, shape: [1, 3, Input.height, Input.width])],
      outputs: [
        "simcc_x": try tensor(simccXData, shape: [1, spec.joints, Input.width * Input.simccSplit]),
        "simcc_y": try tensor(simccYData, shape: [1, spec.joints, Input.height * Input.simccSplit]),
      ],
      runOptions: nil
    )
    let finished = CACurrentMediaTime()
    let points = decode(transform)
    let refined = refine
      ? self.refine(points, in: pixelBuffer)
      : [Double](repeating: 0, count: 2 * FootNet.joints * 3)
    return FootPoseResult(
      points: points,
      refined: refined,
      preprocessMs: (prepared - started) * 1000,
      inferenceMs: (finished - prepared) * 1000,
      refineMs: (CACurrentMediaTime() - finished) * 1000
    )
  }

  // A crop around each foot RTMPose found goes to FootNet, whose 8 points come back in frame coordinates.
  private func refine(_ points: [Double], in pixelBuffer: CVPixelBuffer) -> [Double] {
    let empty = [Double](repeating: 0, count: FootNet.joints * 3)
    guard let footNet else { return empty + empty }
    let width = Double(CVPixelBufferGetWidth(pixelBuffer))
    let height = Double(CVPixelBufferGetHeight(pixelBuffer))
    return Feet.joints.flatMap { foot -> [Double] in
      var seen = foot.indices
        .filter { points[$0 * 3 + 2] >= Feet.minScore }
        .map { (x: points[$0 * 3] * width, y: points[$0 * 3 + 1] * height) }
      let ankle = Feet.ankles[foot.side]
      if seen.count < 2, points[ankle * 3 + 2] >= Feet.minScore {
        seen.append((x: points[ankle * 3] * width, y: points[ankle * 3 + 1] * height))
      }
      guard seen.count >= 2, let box = FootNetRunner.box(around: seen),
            let found = try? footNet.run(pixelBuffer, crop: box), found.count == FootNet.joints * 3 else {
        return empty
      }
      var normalized = [Double]()
      normalized.reserveCapacity(found.count)
      for point in stride(from: 0, to: found.count, by: 3) {
        normalized.append(found[point] / width)
        normalized.append(found[point + 1] / height)
        normalized.append(found[point + 2])
      }
      return normalized
    }
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

  var refine: Bool {
    get { lock.withLock { refineFeet } }
    set { lock.withLock { refineFeet = newValue } }
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
    lock.unlock()
    guard let runner else { throw FootPoseError.notReady(status) }
    guard let frame = frame as? any NativeFrame,
          let sampleBuffer = frame.sampleBuffer,
          let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      throw FootPoseError.unsupportedFrame("no pixel buffer")
    }
    return try runner.run(pixelBuffer, refine: refine)
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
