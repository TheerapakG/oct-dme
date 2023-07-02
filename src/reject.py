import cv2
import numpy as np
from pathlib import Path

from .preprocess import preprocess


def get_reject_score(img):
    prep = preprocess(img)

    blur = cv2.normalize(cv2.bilateralFilter(prep, 32, 150, 150), None, 0, 255, cv2.NORM_MINMAX)  # type: ignore

    _, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big_contours = [c for c in contours if cv2.contourArea(c) > 2500]

    hull = cv2.convexHull(np.vstack(big_contours))

    neighbor_diff = abs(
        cv2.filter2D(prep, -1, np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
    )

    return neighbor_diff.sum() / cv2.contourArea(hull)


if __name__ == "__main__":
    DATA_PATH = Path("./data/dme/")
    # path = Path("./data/dme/Response_f3bc8bc543ac8f076627c71ea8efaf35_L_011.jpg") 19.414806647938477
    # path = Path("./data/dme/Non response_0b9cf0f939daa80b6f7c1457ad104140_L_012.jpg") 48.309320653313506

    for path in DATA_PATH.iterdir():
        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            print("error", f'"{path}"')
            continue

        score = get_reject_score(cv2.cvtColor(img[:496, 496:, :], cv2.COLOR_BGR2GRAY))

        if score > 30:
            print(score, f'"{path}"')
