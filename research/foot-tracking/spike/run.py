import argparse
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .candidates import CANDIDATES, geometric_from_mask
from .common import KEYPOINTS, LABELS, RESULTS, VIEWS, Foot, prepared_images, view_of
from .metrics import labeled_foot, match, summarize

TRUTH_COLOR = (0, 200, 0)
PREDICTION_COLOR = (0, 0, 255)


def draw_feet(canvas: np.ndarray, feet: list[Foot], color) -> None:
    for foot in feet:
        toe, heel = foot.get("big_toe"), foot.get("heel")
        if toe is not None and heel is not None:
            cv2.arrowedLine(canvas, tuple(heel.astype(int)), tuple(toe.astype(int)), color, 3, tipLength=0.08)
        for name in KEYPOINTS:
            point = foot.get(name)
            if point is not None:
                cv2.circle(canvas, tuple(point.astype(int)), 7, color, -1)


def evaluate(names: list[str]) -> dict:
    labels = json.loads(LABELS.read_text())
    images = [p for p in prepared_images() if p.stem in labels]
    if not images:
        raise SystemExit(f"no labeled images: run spike.prepare and spike.label first ({LABELS})")

    report = {}
    for name in names:
        predict = CANDIDATES[name]()
        overlays = RESULTS / "overlays" / name
        overlays.mkdir(parents=True, exist_ok=True)
        per_view = defaultdict(list)
        latencies = []
        for path in images:
            image = cv2.imread(str(path))
            started = time.perf_counter()
            predicted = predict(image, path.stem)
            latencies.append((time.perf_counter() - started) * 1000)
            truth = [labeled_foot(raw) for raw in labels[path.stem]["feet"]]
            per_view[view_of(path.stem)].extend(match(truth, predicted, view_of(path.stem)))
            canvas = image.copy()
            draw_feet(canvas, truth, TRUTH_COLOR)
            draw_feet(canvas, predicted, PREDICTION_COLOR)
            cv2.imwrite(str(overlays / path.name), canvas)
        report[name] = {
            "latency_median_ms_mac": float(np.median(latencies)),
            **{view: summarize(per_view[view]) for view in VIEWS if per_view[view]},
            "all": summarize([r for rs in per_view.values() for r in rs]),
        }
        print(f"{name}: done, overlays in {overlays}")
    return report


def markdown(report: dict) -> str:
    rows = ["| candidate | view | feet | detected | PCK@0.05 toe | PCK@0.05 heel | PCK@0.10 toe | PCK@0.10 heel | axis err, ° | ms (Mac) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, pct=True: "—" if v is None else (f"{v:.0%}" if pct else f"{v:.1f}")
    for name, data in report.items():
        for view in (*VIEWS, "all"):
            s = data.get(view)
            if not s:
                continue
            rows.append(
                f"| {name} | {view} | {s['feet']} | {fmt(s['detection_rate'])} | {fmt(s['pck@0.05_big_toe'])} | "
                f"{fmt(s['pck@0.05_heel'])} | {fmt(s['pck@0.10_big_toe'])} | {fmt(s['pck@0.10_heel'])} | "
                f"{fmt(s['axis_error_median_deg'], pct=False)} | {data['latency_median_ms_mac']:.0f} |"
            )
    return "\n".join(rows)


def self_test() -> None:
    height, width = 900, 700
    mask = np.zeros((height, width), np.uint8)
    angle = np.radians(20)
    toe = np.array([350.0, 150.0])
    direction = np.array([np.sin(angle), -np.cos(angle)])
    heel = toe - direction * 420
    center = (toe + heel) / 2
    cv2.ellipse(mask, tuple(center.astype(int)), (215, 80), float(np.degrees(np.arctan2(direction[1], direction[0]))), 0, 360, 255, -1)
    shin_top = heel + np.array([0, 60])
    cv2.line(mask, tuple(shin_top.astype(int)), (int(shin_top[0]) + 40, height - 1), 255, 150)

    with tempfile.TemporaryDirectory() as tmp:
        mask_dir = Path(tmp)
        cv2.imwrite(str(mask_dir / "top_001.png"), mask)
        predicted = geometric_from_mask(mask_dir)(np.zeros((height, width, 3), np.uint8), "top_001")

    truth = Foot(points={"big_toe": tuple(toe), "heel": tuple(heel), "small_toe": None, "ankle": None})
    results = match([truth], predicted, "top")
    assert results and results[0].detected, "geometric candidate missed the synthetic foot"
    assert results[0].axis_error_deg < 8, f"axis error {results[0].axis_error_deg:.1f}° is above 8°"
    print(f"self-test passed: axis error {results[0].axis_error_deg:.1f}°, toe error {results[0].errors['big_toe']:.1%} of foot length")


def main() -> None:
    parser = argparse.ArgumentParser(description="Foot tracking spike: compare candidates on labeled photos")
    parser.add_argument("--candidates", default=",".join(CANDIDATES), help=f"comma-separated subset of {list(CANDIDATES)}")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    report = evaluate([n.strip() for n in args.candidates.split(",") if n.strip()])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2))
    table = markdown(report)
    (RESULTS / "report.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
