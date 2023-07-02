import cv2
import numpy as np
from pathlib import Path


def preprocess_step1(img):
    # STEP 1: remove big grains
    norm = cv2.normalize(
        img,
        None,  # type: ignore
        0,
        255,
        cv2.NORM_MINMAX,
    )

    neighbor_diff = cv2.normalize(
        abs(cv2.filter2D(norm, -1, np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))),  # type: ignore
        None,  # type: ignore
        0,
        255,
        cv2.NORM_MINMAX,
    )

    neighbor_diff_blur = cv2.normalize(cv2.GaussianBlur(neighbor_diff, (5, 5), 0), None, 0, 255, cv2.NORM_MINMAX)  # type: ignore

    keep = abs(
        cv2.filter2D(
            cv2.normalize(
                abs(norm - neighbor_diff_blur),  # type: ignore
                None,  # type: ignore
                0,
                255,
                cv2.NORM_MINMAX,
            ),
            -1,
            np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),  # type: ignore
        )
    )

    keep_blur = cv2.normalize(cv2.GaussianBlur(keep, (5, 5), 0), None, 0, 255, cv2.NORM_MINMAX)  # type: ignore

    _, thresh = cv2.threshold(keep_blur, 105, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        255 - thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    big_contours = [c for c in contours if cv2.contourArea(c) > 2500]

    stencil = np.zeros(img.shape).astype(img.dtype)
    cv2.fillPoly(stencil, big_contours, (255))
    intermediate = cv2.bitwise_and(norm, stencil)

    _, intermediate_thresh = cv2.threshold(intermediate, 15, 255, cv2.THRESH_BINARY)

    intermediate_contours, _ = cv2.findContours(
        intermediate_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    big_intermediate_contours = [
        c for c in intermediate_contours if cv2.contourArea(c) > 2500
    ]

    # TODO: better thing than this
    hull = cv2.convexHull(np.vstack(big_intermediate_contours))

    stencil = np.zeros(img.shape).astype(img.dtype)
    cv2.fillPoly(stencil, [hull], (255))
    return cv2.bitwise_and(norm, stencil)


def preprocess_step2(img):
    # STEP 2: remove small grains
    norm = cv2.normalize(
        img,
        None,  # type: ignore
        0,
        255,
        cv2.NORM_MINMAX,
    )

    blur = cv2.normalize(cv2.bilateralFilter(norm, 32, 150, 150), None, 0, 255, cv2.NORM_MINMAX)  # type: ignore

    _, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big_contours = [c for c in contours if cv2.contourArea(c) > 2500]

    hull = cv2.convexHull(np.vstack(big_contours))

    stencil = np.zeros(img.shape).astype(img.dtype)
    cv2.fillPoly(stencil, [hull], (255))
    return cv2.bitwise_and(norm, stencil)


def preprocess(img):
    img = preprocess_step1(img)
    return preprocess_step2(img)


if __name__ == "__main__":
    # TODO: cmd line
    # IMG_PATH = Path("./data/dme/Response_f3bc8bc543ac8f076627c71ea8efaf35_L_011.jpg")
    IMG_PATH = Path(
        "./data/dme/Non response_0b9cf0f939daa80b6f7c1457ad104140_L_012.jpg"
    )

    img = cv2.imread(str(IMG_PATH))
    result = preprocess(cv2.cvtColor(img[:496, 496:, :], cv2.COLOR_BGR2GRAY))
    cv2.imshow("result", result)
    cv2.waitKey(10000)
