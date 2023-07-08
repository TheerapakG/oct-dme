import cv2
from dataclasses import replace
import numpy as np
from pathlib import Path

from .utils import Context, imshow, imshow_contours


def preprocess_step1_alt(ctx: Context):
    # WARN: EXPERIMENTAL, NOT WORKING
    if ctx.reject:
        return ctx

    imshow(ctx.img, name="img")

    norm = cv2.normalize(
        ctx.img,
        None,  # type: ignore
        0,
        255,
        cv2.NORM_MINMAX,
    )

    intermediate = norm.copy()
    for _ in range(8):
        _, thresh = cv2.threshold(intermediate, 15, 255, cv2.THRESH_BINARY)

        neighbor = cv2.GaussianBlur(thresh, (7, 7), 0)
        _, neighbor_thresh = cv2.threshold(neighbor, 191, 255, cv2.THRESH_BINARY)

        intermediate = cv2.bitwise_and(intermediate, neighbor_thresh)

    imshow(intermediate)

    _, intermediate_thresh = cv2.threshold(intermediate, 15, 255, cv2.THRESH_BINARY)
    imshow(intermediate_thresh)

    intermediate_contours, _ = cv2.findContours(
        intermediate_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    big_intermediate_contours = [
        c for c in intermediate_contours if cv2.contourArea(c) > 2500
    ]
    imshow_contours(big_intermediate_contours, (0, 0, 255))

    hull = cv2.convexHull(np.vstack(big_intermediate_contours))

    stencil = np.zeros(ctx.img.shape).astype(ctx.img.dtype)
    cv2.fillPoly(stencil, [hull], (255))
    result = cv2.bitwise_and(norm, stencil)
    imshow(result)
    return replace(ctx, img=result)


def preprocess_step1(ctx: Context):
    # STEP 1: remove big grains
    if ctx.reject:
        return ctx

    norm = cv2.normalize(
        ctx.img,
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

    dark_neighbor_diff_blur = cv2.bitwise_and(cv2.bitwise_not(norm), neighbor_diff_blur)
    not_keep = cv2.normalize(
        abs(
            cv2.filter2D(
                dark_neighbor_diff_blur,
                -1,
                np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),  # type: ignore
            )  # now "outlines" remains
        ),
        None,  # type: ignore
        0,
        255,
        cv2.NORM_MINMAX,
    )
    imshow(not_keep)

    not_keep_blur = cv2.normalize(
        cv2.GaussianBlur(not_keep, (5, 5), 0), None, 0, 255, cv2.NORM_MINMAX  # type: ignore
    )
    imshow(not_keep_blur)

    _, thresh = cv2.threshold(not_keep_blur, 75, 255, cv2.THRESH_BINARY_INV)
    # imshow("s1_thresh", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # imshow_contours("s1_contours", contours, (0, 0, 255))
    big_contours = [c for c in contours if cv2.contourArea(c) > 250]
    # imshow_contours("s1_big_contours", big_contours, (0, 0, 255))

    stencil = np.zeros(ctx.img.shape).astype(ctx.img.dtype)
    cv2.fillPoly(stencil, big_contours, (255))
    intermediate = cv2.bitwise_and(norm, stencil)
    imshow(intermediate)

    _, intermediate_thresh = cv2.threshold(intermediate, 15, 255, cv2.THRESH_BINARY)
    imshow(intermediate_thresh)

    intermediate_contours, _ = cv2.findContours(
        intermediate_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    big_intermediate_contours = [
        c for c in intermediate_contours if cv2.contourArea(c) > 2500
    ]
    imshow_contours(big_intermediate_contours, (0, 0, 255))

    hull = cv2.convexHull(np.vstack(big_intermediate_contours))

    stencil = np.zeros(ctx.img.shape).astype(ctx.img.dtype)
    cv2.fillPoly(stencil, [hull], (255))
    result = cv2.bitwise_and(norm, stencil)
    imshow(result)
    return replace(ctx, img=result)


def preprocess_step2(ctx: Context):
    # STEP 2: remove small grains
    if ctx.reject:
        return ctx

    norm = cv2.normalize(
        ctx.img,
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

    stencil = np.zeros(ctx.img.shape).astype(ctx.img.dtype)
    cv2.fillPoly(stencil, [hull], (255))
    result = cv2.bitwise_and(norm, stencil)
    imshow(result)
    return replace(ctx, img=result)


def reject_step2(ctx: Context):
    if ctx.reject:
        return ctx

    norm = cv2.normalize(
        ctx.img,
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

    neighbor_diff = abs(
        cv2.filter2D(norm, -1, np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
    )

    score = neighbor_diff.sum() / cv2.contourArea(hull)
    return replace(ctx, img=norm, reject=score > 30, score=[*ctx.score, int(score)])


def preprocess_step3(ctx: Context):
    # STEP 3: normalize rotation
    if ctx.reject:
        return ctx

    norm = cv2.normalize(
        ctx.img,
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
    x, y, w, h = cv2.boundingRect(hull)

    result = cv2.resize(ctx.img[y : y + h, x : x + w], ctx.img.shape[::-1])
    imshow(result)
    return replace(ctx, img=result)


PREPROCESS_LIST = [
    preprocess_step1,
    preprocess_step2,
    reject_step2,
    preprocess_step3,
]


def preprocess(ctx: Context):
    for f in PREPROCESS_LIST:
        ctx = f(ctx)
    return ctx


if __name__ == "__main__":
    # TODO: cmd line
    IMG_PATHS = [
        Path("./data/dme/Response_45a2a8a4ea7ac0423248f93046523dc0_R_001.jpg"),
        Path("./data/dme/Response_f3bc8bc543ac8f076627c71ea8efaf35_L_012.jpg"),
        Path("./data/dme/Non response_0b9cf0f939daa80b6f7c1457ad104140_L_012.jpg"),
        Path("./data/dme/Response_bff83e94923f91e8af14db5714a937d9_L_001.jpg"),
    ]

    img = cv2.imread(str(IMG_PATHS[3]))
    result = preprocess(
        Context(
            cv2.cvtColor(img[:496, 496:, :], cv2.COLOR_BGR2GRAY),
            dbg=preprocess_step1,
        )
    )
    cv2.waitKey(10000)
