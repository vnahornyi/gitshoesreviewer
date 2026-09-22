"""Sample FIND feet for the Blender renderer (its own Python 3.13 env, no torch): data/synth/shapes/."""

import argparse
from pathlib import Path

import numpy as np

from .find_model import FIND, KEYPOINT_VERTICES, Find

OUT = Path(__file__).resolve().parents[1] / "data/synth/shapes"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    find = Find()
    rng = np.random.default_rng(args.seed)
    # Drop the ankle cap so the renderer can extrude a shin from the open boundary.
    faces = np.delete(find.faces, np.load(FIND / "templ_masked_faces.npy"), axis=0)
    np.savez(OUT / "topology.npz", faces=faces, keypoint_vertices=KEYPOINT_VERTICES)
    for index in range(args.count):
        np.save(OUT / f"{index:04d}.npy", find.sample(rng).vertices.astype(np.float32))
    print(f"wrote {args.count} feet to {OUT}")
