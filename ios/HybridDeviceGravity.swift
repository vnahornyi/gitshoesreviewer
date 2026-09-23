import CoreMotion
import Foundation
import NitroModules

class HybridDeviceGravity: HybridDeviceGravitySpec {
  private let motion = CMMotionManager()
  private let queue = OperationQueue()
  private let lock = NSLock()
  private var latest: [Double] = [0, -1, 0]

  override init() {
    super.init()
    guard motion.isDeviceMotionAvailable else { return }
    motion.deviceMotionUpdateInterval = 1.0 / 30.0
    motion.startDeviceMotionUpdates(to: queue) { [weak self] data, _ in
      guard let self, let gravity = data?.gravity else { return }
      self.lock.lock()
      self.latest = [gravity.x, gravity.y, gravity.z]
      self.lock.unlock()
    }
  }

  deinit {
    motion.stopDeviceMotionUpdates()
  }

  var available: Bool {
    motion.isDeviceMotionAvailable
  }

  func current() throws -> [Double] {
    lock.lock()
    defer { lock.unlock() }
    return latest
  }
}
