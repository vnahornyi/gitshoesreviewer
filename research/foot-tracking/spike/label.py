import argparse
import json

import cv2

from .common import KEYPOINTS, LABELS, prepared_images

WINDOW = "label"
COLORS = {"big_toe": (0, 0, 255), "small_toe": (0, 165, 255), "heel": (255, 0, 0), "ankle": (0, 200, 0)}
CAPTIONS = {"big_toe": "toe", "small_toe": "small toe", "heel": "heel (back of the heel, not the ankle bone)", "ankle": "ankle"}
HELP = "click point | x: not visible | n: next foot | u: undo | s: save image | d: discard image | q: quit"
POINT_SETS = {"all": KEYPOINTS, "toe-heel": ("big_toe", "heel")}
DISCARD = object()


def load() -> dict:
    return json.loads(LABELS.read_text()) if LABELS.exists() else {}


def save(labels: dict) -> None:
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    LABELS.write_text(json.dumps(labels, indent=2))


def draw(image, feet, current):
    canvas = image.copy()
    for foot in feet + [current]:
        for name, point in foot.items():
            if point is not None:
                cv2.circle(canvas, (int(point[0]), int(point[1])), 6, COLORS[name], -1)
                cv2.putText(canvas, CAPTIONS[name].split(" ")[0], (int(point[0]) + 8, int(point[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[name], 2)
    return canvas


def label_image(image, points: tuple[str, ...]):
    feet: list[dict] = []
    current: dict = {}
    state = {"click": None}

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (float(x), float(y))

    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        pending = next((k for k in points if k not in current), None)
        prompt = f"click the {CAPTIONS[pending]}" if pending else "done: n = next foot, s = save image"
        cv2.setWindowTitle(WINDOW, f"foot {len(feet) + 1}: {prompt}  —  {HELP}")
        cv2.imshow(WINDOW, draw(image, feet, current))
        key = cv2.waitKey(30) & 0xFF
        if state["click"] and pending:
            current[pending] = state["click"]
        state["click"] = None
        if key == ord("x") and pending:
            current[pending] = None
        elif key == ord("u") and current:
            current.popitem()
        elif key == ord("n") and current:
            feet.append(current)
            current = {}
        elif key == ord("s"):
            return feet + ([current] if current else [])
        elif key == ord("d"):
            return DISCARD
        elif key == ord("q"):
            return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Click foot keypoints on prepared images")
    parser.add_argument("--points", choices=POINT_SETS, default="all", help="which keypoints to click per foot")
    points = POINT_SETS[parser.parse_args().points]
    labels = load()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for path in prepared_images():
        if path.stem in labels:
            continue
        feet = label_image(cv2.imread(str(path)), points)
        if feet is None:
            break
        labels[path.stem] = {"feet": [], "discarded": True} if feet is DISCARD else {"feet": feet}
        save(labels)
    cv2.destroyAllWindows()
    print(f"{len(labels)} images labeled in {LABELS}")


if __name__ == "__main__":
    main()
