require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "ShoeTryOn"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = "https://github.com/vnahornyi/aishoesreviewer"
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => 17.0 }
  s.source       = { :path => "." }

  s.source_files = ["ios/**/*.{swift}"]

  # The keypoint models are 150 MB and are not in git: `model/` is empty in a fresh clone and the
  # detector stays in its error state until they are there. See the README. The globs tolerate that
  # so `pod install` still succeeds, which is what lets the app build and report the problem itself.
  s.resources    = ["model/*.onnx", "model/*.mlmodelc", "shoes/*.usdz"]

  s.frameworks   = ["AVFoundation", "Accelerate", "CoreML", "RealityKit", "CoreMotion", "Vision", "Metal"]

  load 'nitrogen/generated/ios/ShoeTryOn+autolinking.rb'
  add_nitrogen_files(s)

  s.dependency 'React-jsi'
  s.dependency 'React-callinvoker'
  s.dependency 'VisionCamera'
  s.dependency 'onnxruntime-objc', '1.30.0'
  install_modules_dependencies(s)
end
