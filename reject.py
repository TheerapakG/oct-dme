import cv2
import numpy as np
from pathlib import Path

DATA_PATH = Path("./data/dme/")

for path in DATA_PATH.iterdir():
    name = path.stem.split("_")
    data_bin = (name[0], name[1])
    img = cv2.imread(str(path))
    if img is None or img.size == 0:
        print(f"error loading {path}")
        continue

    cropped = img[:496, 496:, :]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _blur = cv2.bilateralFilter(gray, 32, 150, 150)
    blur = cv2.normalize(_blur, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [c for c in contours if cv2.contourArea(c) > 16 * thresh.shape[0]]
    min_p = dict()
    max_p = dict()
    for c in cs:
        for t in c:
            for p in t:
                if p[0] in min_p:
                    min_p[p[0]] = min(p[1], min_p[p[0]])
                else:
                    min_p[p[0]] = p[1]
                if p[0] in max_p:
                    max_p[p[0]] = max(p[1], max_p[p[0]])
                else:
                    max_p[p[0]] = p[1]
    new_c = np.array(
        [
            *[[p] for p in sorted(min_p.items())],
            *[[p] for p in sorted(max_p.items(), reverse=True)],
        ]
    )
    stencil = np.zeros(gray.shape).astype(gray.dtype)
    cv2.fillPoly(stencil, [new_c], (255))
    result = cv2.bitwise_and(gray, stencil)
    neighbor_diff = abs(
        cv2.filter2D(result, -1, np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
    )
    if neighbor_diff.sum() / cv2.contourArea(new_c) > 25:
        print(path)
