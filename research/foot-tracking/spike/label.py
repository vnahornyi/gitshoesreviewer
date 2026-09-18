import json

import cv2

from .common import KEYPOINTS, LABELS, prepared_images

WINDOW = "label"
COLORS = {"big_toe": (0, 0, 255), "small_toe": (0, 165, 255), "heel": (255, 0, 0), "ankle": (0, 200, 0)}
HELP = "click point | x: not visible | n: next foot | u: undo | s: save image | q: quit"


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
    return canvas


def label_image(image) -> list[dict] | None:
    feet: list[dict] = []
    current: dict = {}
    state = {"click": None}

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (float(x), float(y))

    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        pending = next((k for k in KEYPOINTS if k not in current), None)
        cv2.setWindowTitle(WINDOW, f"foot {len(feet) + 1}: {pending or 'done, press n or s'}  —  {HELP}")
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
        elif key == ord("q"):
            return None


def main() -> None:
    labels = load()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for path in prepared_images():
        if path.stem in labels:
            continue
        feet = label_image(cv2.imread(str(path)))
        if feet is None:
            break
        labels[path.stem] = {"feet": feet}
        save(labels)
    cv2.destroyAllWindows()
    print(f"{len(labels)} images labeled in {LABELS}")


if __name__ == "__main__":
    main()
