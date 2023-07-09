import cv2
from dataclasses import replace
from decimal import Decimal
import numpy as np
from pathlib import Path

from .utils import Context, imshow, imshow_contours


def preprocess_step1(ctx: Context):
    # STEP 1: region based removal
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

    structuring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    _, pic = cv2.threshold(norm, 45, 255, cv2.THRESH_BINARY)
    morph_pic = pic.copy()
    while True:
        new_morph_pic = cv2.morphologyEx(
            morph_pic,
            cv2.MORPH_OPEN,
            structuring,
        )
        if (morph_pic == new_morph_pic).all():
            break
        morph_pic = new_morph_pic
    while True:
        new_morph_pic = cv2.morphologyEx(
            morph_pic,
            cv2.MORPH_CLOSE,
            structuring,
        )
        if (morph_pic == new_morph_pic).all():
            break
        morph_pic = new_morph_pic

    _, black = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV)
    morph_black = black.copy()
    while True:
        new_morph_black = cv2.morphologyEx(
            morph_black,
            cv2.MORPH_OPEN,
            structuring,
        )
        if (morph_black == new_morph_black).all():
            break
        morph_black = new_morph_black

    contours_black, _ = cv2.findContours(
        morph_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    imshow_contours(contours_black, (0, 0, 255), morph_pic)

    floodfill_before = cv2.morphologyEx(
        morph_pic,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    floodfill = floodfill_before.copy()
    cv2.fillPoly(floodfill, contours_black, (255))
    for c in contours_black:
        cv2.floodFill(floodfill, None, c[0][0], (0))

    region = cv2.bitwise_and(
        morph_pic,
        cv2.bitwise_or(
            cv2.bitwise_not(floodfill_before),
            cv2.bitwise_and(floodfill_before, floodfill),
        ),
    )

    if ((morph_pic - region) > 0).sum() > (0.3 * (morph_pic > 0).sum()):
        # maybe flood fill gone wrong
        region = morph_pic

    morph_region = region.copy()
    while True:
        new_morph_region = cv2.morphologyEx(
            morph_region,
            cv2.MORPH_OPEN,
            structuring,
        )
        if (morph_region == new_morph_region).all():
            break
        morph_region = new_morph_region
    while True:
        new_morph_region = cv2.morphologyEx(
            morph_region,
            cv2.MORPH_CLOSE,
            structuring,
        )
        if (morph_region == new_morph_region).all():
            break
        morph_region = new_morph_region
    imshow(morph_region)

    intermediate = cv2.normalize(
        cv2.bitwise_and(norm, morph_region),
        None,  # type: ignore
        0,
        255,
        cv2.NORM_MINMAX,
    )

    neighbor_diff = cv2.normalize(
        abs(cv2.filter2D(intermediate, -1, np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))),  # type: ignore
        None,  # type: ignore
        0,
        255,
        cv2.NORM_MINMAX,
    )

    neighbor_diff_blur = cv2.normalize(cv2.GaussianBlur(neighbor_diff, (5, 5), 0), None, 0, 255, cv2.NORM_MINMAX)  # type: ignore

    dark_neighbor_diff_blur = cv2.bitwise_and(
        cv2.bitwise_not(intermediate), neighbor_diff_blur
    )
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

    keep = cv2.bitwise_and(thresh, morph_region)

    morph_keep = keep.copy()
    while True:
        new_morph_keep = cv2.morphologyEx(
            morph_keep,
            cv2.MORPH_CLOSE,
            structuring,
        )
        if (morph_keep == new_morph_keep).all():
            break
        morph_keep = new_morph_keep
    imshow(morph_keep)

    contours, _ = cv2.findContours(morph_keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # imshow_contours("s1_contours", contours, (0, 0, 255))
    big_contours = [c for c in contours if cv2.contourArea(c) > 250]
    imshow_contours(big_contours, (0, 0, 255))

    hull = cv2.convexHull(np.vstack(big_contours))

    stencil = np.zeros(ctx.img.shape).astype(ctx.img.dtype)
    cv2.fillPoly(stencil, [hull], (255))
    result = cv2.bitwise_and(norm, stencil)
    imshow(result)
    return replace(ctx, img=result)


def preprocess_step2(ctx: Context):
    # STEP 2: remove big grains
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

    structuring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    morph_intermediate_thresh = cv2.morphologyEx(
        intermediate_thresh,
        cv2.MORPH_OPEN,
        structuring,
    )
    while True:
        new_morph_intermediate_thresh = cv2.morphologyEx(
            morph_intermediate_thresh,
            cv2.MORPH_CLOSE,
            structuring,
        )
        if (morph_intermediate_thresh == new_morph_intermediate_thresh).all():
            break
        morph_intermediate_thresh = new_morph_intermediate_thresh
    imshow(morph_intermediate_thresh)

    intermediate_contours, _ = cv2.findContours(
        morph_intermediate_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    big_intermediate_contours = [
        c for c in intermediate_contours if cv2.contourArea(c) > 500
    ]
    imshow_contours(big_intermediate_contours, (0, 0, 255))

    hull = cv2.convexHull(np.vstack(big_intermediate_contours))

    stencil = np.zeros(ctx.img.shape).astype(ctx.img.dtype)
    cv2.fillPoly(stencil, [hull], (255))
    result = cv2.bitwise_and(norm, stencil)
    imshow(result)
    return replace(ctx, img=result)


def preprocess_step3(ctx: Context):
    # STEP 3: remove small grains
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


def reject_step3(ctx: Context):
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


def preprocess_step4(ctx: Context):
    # STEP 4: normalize rotation
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
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    imshow_contours([np.intp(box)], (0, 0, 255))

    angle = 90 * round(rect[2] / 90)
    rad = np.deg2rad(np.abs(angle))
    rect_mat = np.array([[np.cos(rad), np.sin(rad)], [np.sin(rad), np.cos(rad)]])

    rect_rot = (
        tuple(
            np.abs(np.sum(np.array(rect[1]).reshape(1, -1) * rect_mat, axis=-1) * 0.5)
        ),
        rect[1],
        angle,
    )
    box_rot = cv2.boxPoints(rect_rot)

    img_mat = cv2.getAffineTransform(
        box[:3].astype(np.float32), box_rot[:3].astype(np.float32)
    )
    result = cv2.warpAffine(norm, img_mat, tuple(np.intp(np.ceil(np.max(box_rot, axis=0) - np.min(box_rot, axis=0)))))  # type: ignore

    imshow(result)
    return replace(ctx, img=result)


PREPROCESS_LIST = [
    preprocess_step1,
    # preprocess_step2,
    preprocess_step3,
    reject_step3,
    preprocess_step4,
]


def preprocess(ctx: Context):
    for f in PREPROCESS_LIST:
        ctx = f(ctx)
    return ctx


if __name__ == "__main__":
    # TODO: cmd line
    IMG_PATHS = [
        Path("./data/dme/Response_9a88ec32f1c2ffe761e3824aab93c7b8_R_004.jpg"),
        Path("./data/dme/Response_45a2a8a4ea7ac0423248f93046523dc0_R_001.jpg"),
        Path("./data/dme/Response_f3bc8bc543ac8f076627c71ea8efaf35_L_012.jpg"),
        Path("./data/dme/Non response_0b9cf0f939daa80b6f7c1457ad104140_L_012.jpg"),
        Path("./data/dme/Response_bff83e94923f91e8af14db5714a937d9_L_001.jpg"),
        Path("./data/dme/Response_81f40dc0d3e0ec390a4960070fa8e609_L_000.jpg"),
        Path("./data/dme/Response_9c26b6082b53bbfcb23fee3cf17cafcf_L_007.jpg"),
    ]

    img = cv2.imread(str(IMG_PATHS[0]))
    result = preprocess(
        Context(
            cv2.cvtColor(img[:496, 496:, :], cv2.COLOR_BGR2GRAY),
            dbg=preprocess_step1,
        )
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
