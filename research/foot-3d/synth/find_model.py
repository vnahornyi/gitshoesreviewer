"""FIND (Boyne et al., MIT) foot model without pytorch3d: a template mesh plus an MLP displacement field.

The checkpoint holds 8 fitted Foot3D feet (shape, pose and registration latents). New feet come from random
convex blends of those latents with a little noise, and a random overall size.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

FIND = Path(__file__).resolve().parents[1] / "vendor/FOCUS/data/find"
KEYPOINTS = ("big toe", "2nd toe", "3rd toe", "4th toe", "little toe", "heel", "outer extrema", "inner extrema")
WIDTH, LATENT, FOURIER_SCALE = 256, 100, 10


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DisplacementField(nn.Module):
    def __init__(self):
        super().__init__()
        # The original draws B right after torch.manual_seed(1) and sorts its 3 rows by norm; reproduce it exactly.
        fourier = torch.randn((3, WIDTH), generator=torch.Generator().manual_seed(1)) * FOURIER_SCALE
        self.register_buffer("fourier", torch.stack(sorted(fourier, key=lambda row: torch.norm(row, p=2))))
        self.base = _mlp([3 + 2 * WIDTH] + [WIDTH] * 5)
        self.base.append(nn.ReLU())
        self.disp = _mlp([WIDTH + 2 * LATENT, WIDTH, WIDTH, WIDTH, 3])

    def forward(self, points: torch.Tensor, shape: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        projected = 2 * np.pi * points @ self.fourier
        features = self.base(torch.cat([points, torch.sin(projected), torch.cos(projected)], -1))
        latents = torch.cat([shape, pose]).expand(len(points), -1)
        return 0.1 * torch.tanh(self.disp(torch.cat([features, latents], -1)))


@dataclass(frozen=True)
class FootMesh:
    """Vertices in metres in the FIND template frame (x toe-ward, z up, left foot), shared faces."""

    vertices: np.ndarray
    faces: np.ndarray

    @property
    def keypoints(self) -> np.ndarray:
        return self.vertices[KEYPOINT_VERTICES]


class Find:
    def __init__(self):
        state = torch.load(FIND / "model.pth", map_location="cpu", weights_only=False)["state_dict"]
        self.field = DisplacementField()
        self.field.base.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("base.")})
        self.field.disp.load_state_dict({k[9:]: v for k, v in state.items() if k.startswith("mlp_disp.")})
        self.field.eval()
        self.template = state["template_verts"][0].float()
        self.faces = state["template_faces"][0].numpy().astype(np.int32)
        self.shapes = state["shapevec_val.data"].float()
        self.poses = state["posevec_val.data"].float()
        self.scales = state["reg_val.data"][:, 6:9].float()
        # The checkpoint template is centred; template.obj (which keypoints.csv indexes) is not.
        self.offset = TEMPLATE_OBJ.mean(0) - self.template.numpy().mean(0)

    @torch.no_grad()
    def mesh(self, shape: torch.Tensor, pose: torch.Tensor, scale: torch.Tensor) -> FootMesh:
        vertices = (self.template + self.field(self.template, shape, pose)) * scale
        return FootMesh(vertices.numpy() + self.offset, self.faces)

    def fitted(self, index: int) -> FootMesh:
        return self.mesh(self.shapes[index], self.poses[index], self.scales[index])

    def sample(self, rng: np.random.Generator, noise: float = 0.15) -> FootMesh:
        weights = torch.from_numpy(rng.dirichlet(np.full(len(self.shapes), 0.5))).float()
        shape = weights @ self.shapes + noise * torch.randn(LATENT, generator=_torch_rng(rng)) * self.shapes.std(0)
        pose = weights @ self.poses + noise * torch.randn(LATENT, generator=_torch_rng(rng)) * self.poses.std(0)
        scale = weights @ self.scales * float(rng.uniform(0.88, 1.12))
        return self.mesh(shape, pose, scale)


def _torch_rng(rng: np.random.Generator) -> torch.Generator:
    return torch.Generator().manual_seed(int(rng.integers(2**31)))


def _template_obj() -> np.ndarray:
    return np.array([list(map(float, l.split()[1:4])) for l in open(FIND / "template.obj") if l.startswith("v ")])


def _keypoint_vertices() -> np.ndarray:
    rows = {r[0]: r[1:] for r in csv.reader(open(FIND / "keypoints.csv"))}
    names = rows["model"]
    return np.array([int(rows["FIND"][names.index(name)]) for name in KEYPOINTS])


TEMPLATE_OBJ = _template_obj()
KEYPOINT_VERTICES = _keypoint_vertices()
