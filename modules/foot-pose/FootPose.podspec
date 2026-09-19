require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "FootPose"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = "https://github.com/vnahornyi/aishoesreviewer"
  s.license      = package["license"]
  s.authors      = "Vladyslav Nahornyi"

  s.platforms    = { :ios => 17.0 }
  s.source       = { :path => "." }

  s.source_files = ["ios/**/*.{swift}"]
  s.resources    = ["model/*.onnx"]
  s.frameworks   = ["AVFoundation", "Accelerate"]

  load 'nitrogen/generated/ios/FootPose+autolinking.rb'
  add_nitrogen_files(s)

  s.dependency 'React-jsi'
  s.dependency 'React-callinvoker'
  s.dependency 'VisionCamera'
  s.dependency 'onnxruntime-objc', '1.30.0'
  install_modules_dependencies(s)
end
