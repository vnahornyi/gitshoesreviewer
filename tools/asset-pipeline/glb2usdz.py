# /// script
# requires-python = ">=3.11"
# dependencies = ["usd-core==25.11", "numpy"]
# ///
"""Convert a normalized shoe GLB (flattened meshes, PBR metallic-roughness) into a USDZ that RealityKit loads."""

import argparse
import json
import struct
import tempfile
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils

COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
DTYPES = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
IMAGE_SUFFIX = {"image/jpeg": ".jpg", "image/png": ".png"}


class Glb:
    def __init__(self, path: Path):
        data = path.read_bytes()
        magic, _version, _length = struct.unpack_from("<4sII", data, 0)
        if magic != b"glTF":
            raise SystemExit(f"{path} is not a binary glTF")
        offset = 12
        self.json: dict = {}
        self.bin = b""
        while offset < len(data):
            length, kind = struct.unpack_from("<II", data, offset)
            chunk = data[offset + 8 : offset + 8 + length]
            if kind == 0x4E4F534A:
                self.json = json.loads(chunk)
            elif kind == 0x004E4942:
                self.bin = chunk
            offset += 8 + length

    def view(self, index: int) -> bytes:
        view = self.json["bufferViews"][index]
        start = view.get("byteOffset", 0)
        return self.bin[start : start + view["byteLength"]]

    def accessor(self, index: int) -> np.ndarray:
        accessor = self.json["accessors"][index]
        view = self.json["bufferViews"][accessor["bufferView"]]
        components = COMPONENTS[accessor["type"]]
        dtype = np.dtype(DTYPES[accessor["componentType"]])
        stride = view.get("byteStride", components * dtype.itemsize)
        raw = self.view(accessor["bufferView"])
        start = accessor.get("byteOffset", 0)
        count = accessor["count"]
        rows = np.lib.stride_tricks.as_strided(
            np.frombuffer(raw, dtype=np.uint8, offset=start),
            shape=(count, components * dtype.itemsize),
            strides=(stride, 1),
        )
        values = np.ascontiguousarray(rows).view(dtype).reshape(count, components)
        if accessor.get("normalized"):
            values = values.astype(np.float32) / np.iinfo(dtype).max
        return values

    def image(self, index: int) -> tuple[bytes, str]:
        image = self.json["images"][index]
        return self.view(image["bufferView"]), IMAGE_SUFFIX[image["mimeType"]]


def texture_file(glb: Glb, texture_index: int, folder: Path, written: dict[int, str]) -> str:
    image_index = glb.json["textures"][texture_index]["source"]
    if image_index not in written:
        data, suffix = glb.image(image_index)
        name = f"texture_{image_index}{suffix}"
        (folder / name).write_bytes(data)
        written[image_index] = name
    return written[image_index]


def add_texture(stage, material_path: str, name: str, file: str, st_reader, raw: bool):
    shader = UsdShade.Shader.Define(stage, f"{material_path}/{name}")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{file}")
    shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
    shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if raw else "sRGB")
    return shader


def define_material(stage, glb: Glb, index: int, folder: Path, written: dict[int, str]) -> UsdShade.Material:
    source = glb.json["materials"][index]
    pbr = source.get("pbrMetallicRoughness", {})
    path = f"/Shoe/Materials/Material_{index}"
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, f"{path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

    st_reader = UsdShade.Shader.Define(stage, f"{path}/StReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")

    base_factor = pbr.get("baseColorFactor", [1, 1, 1, 1])
    if "baseColorTexture" in pbr:
        file = texture_file(glb, pbr["baseColorTexture"]["index"], folder, written)
        texture = add_texture(stage, path, "BaseColor", file, st_reader, raw=False)
        texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*base_factor))
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    else:
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*base_factor[:3]))

    metallic = pbr.get("metallicFactor", 1.0)
    roughness = pbr.get("roughnessFactor", 1.0)
    if "metallicRoughnessTexture" in pbr:
        file = texture_file(glb, pbr["metallicRoughnessTexture"]["index"], folder, written)
        texture = add_texture(stage, path, "MetallicRoughness", file, st_reader, raw=True)
        texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1, roughness, metallic, 1))
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(texture.ConnectableAPI(), "g")
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(texture.ConnectableAPI(), "b")
    else:
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)

    if "normalTexture" in source:
        file = texture_file(glb, source["normalTexture"]["index"], folder, written)
        texture = add_texture(stage, path, "Normal", file, st_reader, raw=True)
        texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
        texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        surface.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(texture.ConnectableAPI(), "rgb")

    if "occlusionTexture" in source:
        file = texture_file(glb, source["occlusionTexture"]["index"], folder, written)
        texture = add_texture(stage, path, "Occlusion", file, st_reader, raw=True)
        surface.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(texture.ConnectableAPI(), "r")

    return material


def define_mesh(stage, glb: Glb, name: str, primitive: dict, material, mirror: bool) -> None:
    attributes = primitive["attributes"]
    positions = glb.accessor(attributes["POSITION"]).astype(np.float32)
    indices = (
        glb.accessor(primitive["indices"]).reshape(-1).astype(np.int32)
        if "indices" in primitive
        else np.arange(len(positions), dtype=np.int32)
    )
    if mirror:
        positions[:, 0] *= -1
        indices = indices.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1)
    mesh = UsdGeom.Mesh.Define(stage, f"/Shoe/Geometry/{name}")
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreatePointsAttr(positions.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
    mesh.CreateFaceVertexIndicesAttr(indices.tolist())
    # glTF winds counter-clockwise like USD's default right-handed orientation.
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateExtentAttr([positions.min(axis=0).tolist(), positions.max(axis=0).tolist()])
    if "NORMAL" in attributes:
        normals = glb.accessor(attributes["NORMAL"]).astype(np.float32).copy()
        if mirror:
            normals[:, 0] *= -1
        mesh.CreateNormalsAttr(normals.tolist())
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if "TEXCOORD_0" in attributes:
        uv = glb.accessor(attributes["TEXCOORD_0"]).astype(np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        primvar.Set(uv.tolist())
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def convert(source: Path, target: Path, mirror: bool) -> None:
    glb = Glb(source)
    if glb.json.get("extensionsRequired"):
        raise SystemExit(f"unsupported required extensions: {glb.json['extensionsRequired']}")

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        layer = folder / "model.usdc"
        stage = Usd.Stage.CreateNew(str(layer))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Shoe")
        stage.SetDefaultPrim(root.GetPrim())

        written: dict[int, str] = {}
        materials = {
            index: define_material(stage, glb, index, folder, written)
            for index in range(len(glb.json.get("materials", [])))
        }
        for mesh_index, mesh in enumerate(glb.json.get("meshes", [])):
            for primitive_index, primitive in enumerate(mesh["primitives"]):
                material = materials.get(primitive.get("material"))
                define_mesh(stage, glb, f"Mesh_{mesh_index}_{primitive_index}", primitive, material, mirror)

        stage.GetRootLayer().Save()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(layer)), str(target)):
            raise SystemExit("usdz packaging failed")
    print(f"wrote {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="normalized .glb from normalize.mjs")
    parser.add_argument("target", type=Path, help=".usdz to write")
    parser.add_argument("--mirror", action="store_true", help="mirror across x to turn a right shoe into a left one, or back")
    args = parser.parse_args()
    convert(args.source, args.target, args.mirror)


if __name__ == "__main__":
    main()
