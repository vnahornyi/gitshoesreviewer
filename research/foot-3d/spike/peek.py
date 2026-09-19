import sys
import time

import cv2
import numpy as np

from .toc import load_model, predict

device = "mps"
model = load_model(device)
for path in sys.argv[1:]:
    image = cv2.imread(path)
    started = time.perf_counter()
    p = predict(model, image, device)
    ms = (time.perf_counter() - started) * 1000
    inside = p.toc[p.mask]
    print(f"{path.split('/')[-1]}: {ms:.0f} ms, mask {p.mask.mean():.1%}, foot {p.footedness}, "
          f"toc min {inside.min(0).round(2) if len(inside) else '-'} max {inside.max(0).round(2) if len(inside) else '-'}, "
          f"std median {np.median(p.toc_std[p.mask]) if p.mask.any() else '-'}")
