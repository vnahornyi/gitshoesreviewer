"""Render synthetic feet with Blender as a library (bpy), no Blender UI.

Each frame: a left and a right FIND foot with shins, bare or in socks, often under trousers, on a random floor in
front of a random wall, seen as in a mirror, from above (looking down at your own feet) or by another person.
Outputs per frame: <id>.jpg, <id>_mask.png (red = left foot, green = right foot, visible pixels only) and <id>.json
(camera intrinsics and pose, each foot's 4×4 pose in the FIND template frame, its 8 keypoints and their visibility).

    .venv/bin/python render.py --start 0 --count 20

Every frame compiles new Metal shaders and the driver's shader heap is never freed, so one process dies after ~900
frames. Without --count the script renders in child processes of BATCH frames each and skips frames already on disk.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run as a script, not with -m: the project root is not on the path

from assets import ASSETS  # noqa: E402
SHAPES = ASSETS / "synth/shapes"
OUT = ASSETS / "synth/renders"
WIDTH, HEIGHT = 720, 960
SAMPLES = 24
SHIN_TOP = 0.95
BATCH = 200
MASK_COLORS = {"left": (1.0, 0.0, 0.0), "right": (0.0, 1.0, 0.0)}


def reset(device: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    if device == "GPU":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "METAL"
        preferences.get_devices()
        for d in preferences.devices:
            d.use = d.type == "METAL"
        scene.cycles.device = "GPU"
    scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
    scene.render.resolution_percentage = 100
    scene.world = bpy.data.worlds.new("world")
    scene.world.use_nodes = True
    return scene


# ---------- geometry ----------

def boundary_loop(faces: np.ndarray) -> np.ndarray:
    """The longest open boundary of the foot mesh: the ankle opening."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    following = {}
    for a, b in unique[counts == 1]:
        following.setdefault(a, []).append(b)
        following.setdefault(b, []).append(a)
    loops, seen = [], set()
    for start in following:
        if start in seen:
            continue
        loop, previous, current = [start], None, start
        seen.add(start)
        while True:
            options = [v for v in following[current] if v != previous and (v not in seen or v == start)]
            if not options or options[0] == start:
                break
            previous, current = current, options[0]
            loop.append(current)
            seen.add(current)
        loops.append(loop)
    return np.array(max(loops, key=len))


def shin_mesh(vertices: np.ndarray, loop: np.ndarray, rng: np.random.Generator):
    """Extrude the ankle opening up into a tapered, slightly bent shin; the first ring is the foot's own boundary."""
    ring = vertices[loop]
    centre = ring.mean(0)
    lean = np.array([rng.uniform(-0.12, 0.05), rng.uniform(-0.05, 0.05), 1.0])
    heights = np.linspace(0, SHIN_TOP - centre[2], 14)
    rings = []
    for i, h in enumerate(heights):
        t = h / heights[-1]
        target_radius = 0.042 + 0.018 * math.sin(math.pi * min(1.0, t * 1.6)) + 0.03 * max(0.0, t - 0.55) / 0.45
        offsets = (ring - centre) * [1, 1, 0]
        radius = np.linalg.norm(offsets, axis=1).mean()
        blend = min(1.0, i / 2)
        scale = (1 - blend) + blend * target_radius / radius
        rings.append(ring * [1, 1, 1 - blend] + [0, 0, blend * centre[2]] + lean * h + (offsets * (scale - 1)))
    points = np.concatenate(rings)
    n = len(ring)
    faces = []
    for r in range(len(rings) - 1):
        for k in range(n):
            a, b = r * n + k, r * n + (k + 1) % n
            faces.append((a, b, b + n, a + n))
    faces.append(tuple(range(len(points) - 1, len(points) - n - 1, -1)))
    return points, faces


def make_object(name: str, points, faces, material, mirror: bool):
    points = np.array(points, dtype=float)
    faces = [tuple(f) for f in faces]
    if mirror:
        points = points * [1, -1, 1]
        faces = [f[::-1] for f in faces]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points.tolist(), [], faces)
    mesh.update()
    mesh.shade_smooth()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)
    return obj


# ---------- materials ----------

def principled(name: str):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    return material, nodes, nodes["Principled BSDF"], material.node_tree.links


def bump(nodes, links, bsdf, scale: float, strength: float, kind: str = "noise"):
    texture = nodes.new("ShaderNodeTexNoise" if kind == "noise" else "ShaderNodeTexWave")
    texture.inputs["Scale"].default_value = scale
    node = nodes.new("ShaderNodeBump")
    node.inputs["Strength"].default_value = strength
    links.new(texture.outputs["Fac"] if kind == "noise" else texture.outputs["Color"], node.inputs["Height"])
    links.new(node.outputs["Normal"], bsdf.inputs["Normal"])
    return texture


def skin_color(rng) -> tuple:
    light, dark = np.array([0.85, 0.62, 0.52]), np.array([0.28, 0.16, 0.10])
    c = light + rng.uniform(0, 1) ** 1.5 * (dark - light)
    return (*(c * rng.uniform(0.9, 1.1, 3)).clip(0, 1), 1)


def random_color(rng, saturation=(0, 1), value=(0.05, 0.95)) -> tuple:
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(rng.uniform(), rng.uniform(*saturation), rng.uniform(*value))
    return (r, g, b, 1)


def leg_material(rng, skin, sock: dict | None):
    """Skin, with a sock below `sock["height"]` (object-space z, so it follows the foot)."""
    material, nodes, bsdf, links = principled("leg")
    bsdf.inputs["Base Color"].default_value = skin
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.35, 0.6)
    bsdf.inputs["Subsurface Weight"].default_value = rng.uniform(0.05, 0.25)
    bump(nodes, links, bsdf, rng.uniform(200, 600), 0.05)
    if sock is None:
        return material
    cloth = nodes.new("ShaderNodeBsdfPrincipled")
    cloth.inputs["Roughness"].default_value = rng.uniform(0.75, 1.0)
    cloth.inputs["Sheen Weight"].default_value = rng.uniform(0.2, 0.8)
    if rng.random() < 0.35:
        stripes = nodes.new("ShaderNodeTexWave")
        stripes.inputs["Scale"].default_value = rng.uniform(4, 25)
        stripes.wave_type = "BANDS"
        stripes.bands_direction = rng.choice(["X", "Y", "Z"])
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.interpolation = "CONSTANT"
        ramp.color_ramp.elements[0].color = sock["color"]
        ramp.color_ramp.elements[1].position = 0.5
        ramp.color_ramp.elements[1].color = random_color(rng)
        links.new(stripes.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], cloth.inputs["Base Color"])
    else:
        cloth.inputs["Base Color"].default_value = sock["color"]
    knit = nodes.new("ShaderNodeTexWave")
    knit.inputs["Scale"].default_value = rng.uniform(150, 400)
    knit_bump = nodes.new("ShaderNodeBump")
    knit_bump.inputs["Strength"].default_value = rng.uniform(0.1, 0.4)
    links.new(knit.outputs["Fac"], knit_bump.inputs["Height"])
    links.new(knit_bump.outputs["Normal"], cloth.inputs["Normal"])

    coordinates = nodes.new("ShaderNodeTexCoord")
    split = nodes.new("ShaderNodeSeparateXYZ")
    below = nodes.new("ShaderNodeMath")
    below.operation = "LESS_THAN"
    below.inputs[1].default_value = sock["height"]
    mix = nodes.new("ShaderNodeMixShader")
    links.new(coordinates.outputs["Object"], split.inputs["Vector"])
    links.new(split.outputs["Z"], below.inputs[0])
    links.new(below.outputs["Value"], mix.inputs["Fac"])
    links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    links.new(cloth.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], nodes["Material Output"].inputs["Surface"])
    return material


def trousers(rng, shin_points: np.ndarray, ring_size: int, mirror: bool, material):
    """A tube around the shin from the hem up, following the shin's axis, loose and slightly wrinkled."""
    hem = rng.uniform(0.06, 0.35)
    top = SHIN_TOP
    axis = shin_points.reshape(-1, ring_size, 3).mean(1)
    radius = rng.uniform(0.078, 0.11)
    segments, rings = 48, 12
    points, faces = [], []
    for r in range(rings):
        z = hem + (top - hem) * r / (rings - 1)
        t = r / (rings - 1)
        flare = radius * (1 + rng.uniform(-0.05, 0.1) * (1 - t) + 0.6 * max(0.0, t - 0.5))
        centre = [np.interp(z, axis[:, 2], axis[:, 0]), np.interp(z, axis[:, 2], axis[:, 1])]
        for k in range(segments):
            angle = 2 * math.pi * k / segments
            wobble = 1 + 0.06 * math.sin(3 * angle + r) * rng.uniform(0.3, 1)
            points.append((centre[0] + flare * wobble * math.cos(angle), centre[1] + flare * wobble * math.sin(angle), z))
    for r in range(rings - 1):
        for k in range(segments):
            a, b = r * segments + k, r * segments + (k + 1) % segments
            faces.append((a, b, b + segments, a + segments))
    faces.append(tuple(range(len(points) - 1, len(points) - segments - 1, -1)))
    obj = make_object("trousers", points, faces, material, mirror)
    solidify = obj.modifiers.new("thickness", "SOLIDIFY")
    solidify.thickness = 0.004
    return obj


def cloth_material(rng):
    material, nodes, bsdf, links = principled("trousers")
    denim = rng.random() < 0.4
    bsdf.inputs["Base Color"].default_value = (
        (rng.uniform(0.02, 0.1), rng.uniform(0.05, 0.15), rng.uniform(0.15, 0.4), 1) if denim else random_color(rng, (0, 0.6), (0.02, 0.7))
    )
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.6, 0.95)
    bump(nodes, links, bsdf, rng.uniform(80, 300), 0.2, "wave" if denim else "noise")
    return material


def floor_material(rng):
    material, nodes, bsdf, links = principled("floor")
    kind = rng.choice(["wood", "tiles", "noise", "carpet"])
    if kind == "wood":
        texture = nodes.new("ShaderNodeTexWave")
        texture.inputs["Scale"].default_value = rng.uniform(1, 6)
        texture.inputs["Distortion"].default_value = rng.uniform(2, 12)
    elif kind == "tiles":
        texture = nodes.new("ShaderNodeTexBrick" if rng.random() < 0.5 else "ShaderNodeTexChecker")
        texture.inputs["Scale"].default_value = rng.uniform(1, 5)
    else:
        texture = nodes.new("ShaderNodeTexNoise")
        texture.inputs["Scale"].default_value = rng.uniform(5, 200 if kind == "carpet" else 30)
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = random_color(rng, (0, 0.7), (0.05, 0.9))
    ramp.color_ramp.elements[1].color = random_color(rng, (0, 0.7), (0.05, 0.9))
    links.new(texture.outputs["Fac"] if "Fac" in texture.outputs else texture.outputs[0], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.2, 0.95) if kind != "carpet" else 1.0
    return material


def plane(name: str, size: float, material, location=(0, 0, 0), rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_plane_add(size=size, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def lights(rng, scene, target: Vector):
    background = scene.world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = random_color(rng, (0, 0.3), (0.3, 1))
    background.inputs["Strength"].default_value = rng.uniform(0.05, 0.6)
    for i in range(int(rng.integers(1, 4))):
        kind = rng.choice(["AREA", "SUN", "POINT"])
        data = bpy.data.lights.new(f"light{i}", kind)
        data.color = random_color(rng, (0, 0.25), (0.85, 1))[:3]
        if kind == "SUN":
            data.energy = rng.uniform(1, 5)
            data.angle = rng.uniform(0.05, 0.5)
        else:
            data.energy = rng.uniform(150, 2000)
            if kind == "AREA":
                data.size = rng.uniform(0.5, 3)
        obj = bpy.data.objects.new(f"light{i}", data)
        bpy.context.collection.objects.link(obj)
        direction = Vector(rng.normal(0, 1, 3)).normalized()
        direction.z = abs(direction.z) + 0.4
        obj.location = target + direction.normalized() * rng.uniform(1.5, 4)
        obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


# ---------- pose and camera ----------

def foot_pose(rng, vertices: np.ndarray, yaw: float, offset: np.ndarray) -> Matrix:
    """Foot on the floor with a random yaw; sometimes the heel is raised or the whole foot is lifted."""
    pitch = rng.uniform(0.1, 0.6) if rng.random() < 0.25 else 0.0
    toe_ball = vertices[:, 0].max() - 0.07
    raise_heel = Matrix.Translation((toe_ball, 0, 0)) @ Matrix.Rotation(pitch, 4, "Y") @ Matrix.Translation((-toe_ball, 0, 0))
    pose = Matrix.Rotation(yaw, 4, "Z") @ raise_heel
    lowest = min((pose @ Vector(v)).z for v in vertices[::25])
    lift = rng.uniform(0.02, 0.12) if rng.random() < 0.1 else 0.0
    return Matrix.Translation((offset[0], offset[1], -lowest + lift)) @ pose


def place_camera(rng, scene, target: Vector, facing: float):
    mode = rng.choice(["mirror", "top", "third"], p=[0.5, 0.25, 0.25])
    if mode == "mirror":
        azimuth = facing + (rng.uniform(-1.2, 1.2) if rng.random() < 0.9 else math.pi + rng.uniform(-0.8, 0.8))
        distance, height = rng.uniform(0.8, 2.6), rng.uniform(0.6, 1.5)
    elif mode == "top":
        # Looking down at your own feet: the phone is at chest height, held out over the toes.
        azimuth = facing + rng.uniform(-0.6, 0.6)
        distance, height = rng.uniform(0.15, 0.5), rng.uniform(1.0, 1.4)
    else:
        azimuth = rng.uniform(0, 2 * math.pi)
        distance, height = rng.uniform(0.6, 2.0), rng.uniform(0.2, 1.1)
    location = target + Vector((math.cos(azimuth) * distance, math.sin(azimuth) * distance, height))
    data = bpy.data.cameras.new("camera")
    data.sensor_fit = "VERTICAL"
    # Mostly the iPhone wide lens, sometimes zoomed in the way a crop of a mirror shot ends up.
    data.angle = math.radians(rng.uniform(55, 72) if rng.random() < 0.5 else rng.uniform(18, 55))
    camera = bpy.data.objects.new("camera", data)
    bpy.context.collection.objects.link(camera)
    aim = target + Vector(rng.normal(0, 0.05, 3)) + Vector((0, 0, 0.04))
    camera.location = location
    camera.rotation_euler = (aim - location).to_track_quat("-Z", "Y").to_euler()
    camera.rotation_euler.rotate_axis("Z", rng.normal(0, 0.12))
    scene.camera = camera
    return camera, mode


def intrinsics(camera) -> list:
    focal = HEIGHT / 2 / math.tan(camera.data.angle / 2)
    return [[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1]]


def keypoint_labels(scene, camera, foot, keypoints_local: np.ndarray) -> dict:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points, visible = [], []
    origin = camera.matrix_world.translation
    for local in keypoints_local:
        world = foot.matrix_world @ Vector(local)
        ndc = world_to_camera_view(scene, camera, world)
        points.append([ndc.x * WIDTH, (1 - ndc.y) * HEIGHT])
        direction = world - origin
        hit, location, _, _, obj, _ = scene.ray_cast(depsgraph, origin, direction.normalized(), distance=direction.length + 0.01)
        visible.append(bool(hit and obj.name == foot.name and (location - world).length < 0.008))
    return {"keypoints": points, "visible": visible}


def render_mask(scene, path: Path, feet: dict):
    """Flat object colours with Workbench, no anti-aliasing: red left foot, green right foot, black elsewhere."""
    for obj in scene.objects:
        obj.color = (0, 0, 0, 1)
    for side, obj in feet.items():
        obj.color = (*MASK_COLORS[side], 1)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.render_aa = "OFF"
    shading = scene.display.shading
    shading.light = "FLAT"
    shading.color_type = "OBJECT"
    shading.background_type = "VIEWPORT"
    shading.background_color = (0, 0, 0)
    scene.world.color = (0, 0, 0)
    scene.view_settings.view_transform = "Standard"
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_frame(index: int, device: str):
    rng = np.random.default_rng(index)
    scene = reset(device)
    topology = np.load(SHAPES / "topology.npz")
    faces, keypoint_vertices = topology["faces"], topology["keypoint_vertices"]
    loop = boundary_loop(faces)
    shapes = sorted(SHAPES.glob("[0-9]*.npy"))

    skin = skin_color(rng)
    sock = {"color": random_color(rng), "height": rng.uniform(0.06, 0.3)} if rng.random() < 0.6 else None
    wear_trousers = rng.random() < 0.7
    cloth = cloth_material(rng)

    facing = rng.uniform(0, 2 * math.pi)
    medial = np.array([-math.sin(facing), math.cos(facing)])
    gap = rng.uniform(0.1, 0.32)
    feet, poses, labels = {}, {}, {}
    vertices = np.load(shapes[rng.integers(len(shapes))]).astype(float)
    medial_sign = 1 if vertices[keypoint_vertices[0], 1] > vertices[keypoint_vertices[4], 1] else -1
    for side in ("left", "right"):
        mirror = side == "right"
        material = leg_material(rng, skin, sock)
        foot = make_object(f"foot_{side}", vertices, faces, material, mirror)
        points, shin_faces = shin_mesh(vertices, loop, rng)
        shin = make_object(f"shin_{side}", points, shin_faces, material, mirror)
        parts = [foot, shin]
        if wear_trousers:
            parts.append(trousers(rng, points, len(loop), mirror, cloth))
        shift = (0 if side == "left" else gap) * medial * medial_sign + rng.normal(0, 0.04, 2)
        yaw = facing + rng.normal(0, 0.25) + (0 if side == "left" else 0)
        pose = foot_pose(rng, vertices * ([1, -1, 1] if mirror else 1), yaw, shift)
        for part in parts:
            part.matrix_world = pose
        feet[side], poses[side] = foot, pose

    target = (feet["left"].matrix_world.translation + feet["right"].matrix_world.translation) / 2
    plane("floor", 12, floor_material(rng))
    wall, _, _, _ = principled("wall")
    wall.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = random_color(rng, (0, 0.4), (0.2, 0.95))
    camera, mode = place_camera(rng, scene, target, facing)
    behind = target + (target - camera.location).normalized() * rng.uniform(0.6, 2.5) * Vector((1, 1, 0))
    plane("wall", 12, wall, location=(behind.x, behind.y, 3), rotation=(math.pi / 2, 0, math.atan2(behind.y - target.y, behind.x - target.x) + math.pi / 2))
    lights(rng, scene, target)
    bpy.context.view_layer.update()

    keypoints_local = vertices[keypoint_vertices]
    for side, foot in feet.items():
        local = keypoints_local * ([1, -1, 1] if side == "right" else 1)
        labels[side] = {"pose": [list(row) for row in poses[side]], **keypoint_labels(scene, camera, foot, local)}

    name = f"{index:06d}"
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 92
    scene.render.filepath = str(OUT / f"{name}.jpg")
    bpy.ops.render.render(write_still=True)
    render_mask(scene, OUT / f"{name}_mask.png", feet)
    (OUT / f"{name}.json").write_text(json.dumps({
        "mode": mode,
        "camera": {"K": intrinsics(camera), "world_from_camera": [list(row) for row in camera.matrix_world]},
        "sock": sock is not None,
        "trousers": wear_trousers,
        "feet": labels,
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, help="render in this process; omit to render --total in batches")
    parser.add_argument("--total", type=int, default=20000)
    parser.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.count is None:
        for start in range(args.start, args.start + args.total, BATCH):
            count = min(BATCH, args.start + args.total - start)
            subprocess.run([sys.executable, __file__, "--start", str(start), "--count", str(count), "--device", args.device])
    else:
        for index in range(args.start, args.start + args.count):
            if not (OUT / f"{index:06d}.json").exists():
                render_frame(index, args.device)
