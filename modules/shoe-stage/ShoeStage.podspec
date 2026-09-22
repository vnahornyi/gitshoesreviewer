require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "ShoeStage"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = "https://github.com/vnahornyi/aishoesreviewer"
  s.license      = package["license"]
  s.authors      = "Vladyslav Nahornyi"

  s.platforms    = { :ios => 17.0 }
  s.source       = { :path => "." }

  s.source_files = ["ios/**/*.{swift}"]
  s.resources    = ["shoes/*.usdz"]
  s.frameworks   = ["RealityKit", "CoreMotion", "Vision", "Metal"]

  load 'nitrogen/generated/ios/ShoeStage+autolinking.rb'
  add_nitrogen_files(s)

  s.dependency 'React-jsi'
  s.dependency 'React-callinvoker'
  s.dependency 'VisionCamera'
  install_modules_dependencies(s)
end
