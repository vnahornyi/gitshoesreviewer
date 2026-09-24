import Foundation

@main
struct DecoderProbe {
  static func main() {
    let bytes = FileHandle.standardInput.readDataToEndOfFile()
    let floatsPerSample = FootNetDecoder.joints * FootNetDecoder.size * FootNetDecoder.size
    guard bytes.count.isMultiple(of: MemoryLayout<Float>.size),
          bytes.count / MemoryLayout<Float>.size % floatsPerSample == 0 else {
      fputs("input must contain complete 8×256×256 float32 heatmap batches\n", stderr)
      exit(2)
    }
    let values = bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    values.withUnsafeBufferPointer { buffer in
      let sampleCount = values.count / floatsPerSample
      for sample in 0..<sampleCount {
        let maps = buffer.baseAddress!.advanced(by: sample * floatsPerSample)
        let points = FootNetDecoder.decode(maps, crop: Crop(x: 0, y: 0, side: Double(FootNetDecoder.size)))
        print(points.map { String(format: "%.9f", $0) }.joined(separator: ","))
      }
    }
  }
}
