import cv2
from dataclasses import replace
import numpy as np
from pathlib import Path

from .utils import Context, imshow, imshow_contours, log, dbg


def morph_opens(img: cv2.Mat, structuring):
    while True:
        new_img = cv2.morphologyEx(
            img,
            cv2.MORPH_OPEN,
            structuring,
        )
        if (img == new_img).all():
            return img
        img = new_img


def morph_closes(img: cv2.Mat, structuring):
    while True:
        new_img = cv2.morphologyEx(
            img,
            cv2.MORPH_CLOSE,
            structuring,
        )
        if (img == new_img).all():
            return img
        img = new_img


def normalize(img: cv2.Mat):
    return cv2.normalize(
        img,
        None,  # type: ignore
        0,
        255,
        cv2.NORM_MINMAX,
    )


def get_hull(mask, size_threshold):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big_contours = [c for c in contours if cv2.contourArea(c) > size_threshold]
    imshow_contours(big_contours, (0, 0, 255))

    return cv2.convexHull(np.vstack(big_contours))


def apply_mask(img, mask, size_threshold):
    stencil = np.zeros(img.shape).astype(img.dtype)
    cv2.fillPoly(stencil, [get_hull(mask, size_threshold)], (255))
    result = cv2.bitwise_and(img, img, mask=stencil)
    imshow(result)

    return result


def preprocess_step1(ctx: Context):
    # STEP 1: region based removal
    if ctx.reject:
        return ctx

    imshow()

    norm = normalize(ctx.img)

    structuring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    _, pic = cv2.threshold(norm, 45, 255, cv2.THRESH_BINARY)
    morph_pic = morph_closes(morph_opens(pic, structuring), structuring)

    return replace(ctx, mask=morph_pic)


def preprocess_step2(ctx: Context):
    # STEP 2: image edge based removal
    if ctx.reject:
        return ctx

    imshow()

    norm = normalize(ctx.img)

    structuring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    _, black = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV)
    morph_black = morph_opens(black, structuring)

    contours_black, _ = cv2.findContours(
        morph_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    imshow_contours(contours_black, (255, 0, 0))

    floodfill_before = cv2.morphologyEx(
        ctx.mask_,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    floodfill = floodfill_before.copy()
    cv2.fillPoly(floodfill, contours_black, (255))
    for c in contours_black:
        cv2.floodFill(floodfill, None, c[0][0], (0))

    region = cv2.bitwise_and(
        ctx.mask_,
        cv2.bitwise_or(
            cv2.bitwise_not(floodfill_before),
            cv2.bitwise_and(floodfill_before, floodfill),
        ),
    )
    morph_region = morph_closes(region, structuring)

    imshow(morph_region)

    if ((ctx.mask_ - morph_region) > 0).sum() < (0.3 * (ctx.mask_ > 0).sum()):  # type: ignore
        return replace(ctx, mask=morph_region)

    # maybe flood fill gone wrong
    return ctx


def preprocess_step3(ctx: Context):
    # STEP 3: coarse grain removal via neighbor diff
    if ctx.reject:
        return ctx

    imshow()

    norm = normalize(ctx.masked)

    structuring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    neighbor_diff = cv2.convertScaleAbs(cv2.Laplacian(norm, cv2.CV_16S))

    neighbor_diff_blur = normalize(cv2.GaussianBlur(neighbor_diff, (5, 5), 0))

    dark_neighbor_diff_blur = cv2.bitwise_and(cv2.bitwise_not(norm), neighbor_diff_blur)
    not_keep = cv2.convertScaleAbs(cv2.Laplacian(dark_neighbor_diff_blur, cv2.CV_16S))
    imshow(not_keep)

    not_keep_blur = cv2.GaussianBlur(not_keep, (5, 5), 0)
    imshow(not_keep_blur)

    _, thresh = cv2.threshold(not_keep_blur, 90, 255, cv2.THRESH_BINARY_INV)

    keep = cv2.bitwise_and(thresh, ctx.mask_)
    morph_keep = morph_closes(keep, structuring)
    imshow(morph_keep)

    return replace(ctx, mask=morph_keep)


def preprocess_step4(ctx: Context):
    # STEP 4: apply mask
    return replace(ctx, img=apply_mask(ctx.img, ctx.mask_, 500), mask=None)


def preprocess_step5(ctx: Context):
    # STEP 5: remove small grains
    if ctx.reject:
        return ctx

    imshow()

    norm = normalize(ctx.masked)

    structuring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    blur = normalize(cv2.bilateralFilter(norm, 32, 150, 150))

    _, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)

    keep = cv2.bitwise_and(thresh, ctx.mask_)
    morph_keep = morph_closes(keep, structuring)
    imshow(morph_keep)

    return replace(ctx, mask=morph_keep)


def preprocess_step6(ctx: Context):
    # STEP 6: apply mask
    return replace(ctx, img=apply_mask(ctx.img, ctx.mask_, 2500), mask=None)


def reject_step6(ctx: Context):
    if ctx.reject:
        return ctx

    norm = normalize(ctx.masked)

    structuring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    blur = normalize(cv2.bilateralFilter(norm, 32, 150, 150))

    _, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)

    keep = cv2.bitwise_and(thresh, ctx.mask_)
    morph_keep = morph_closes(keep, structuring)
    imshow(morph_keep)

    hull = get_hull(morph_keep, 2500)

    neighbor_diff = cv2.convertScaleAbs(cv2.Laplacian(norm, cv2.CV_16S))

    score = neighbor_diff.sum() / cv2.contourArea(hull)
    log.debug(f"{score}")

    return replace(ctx, reject=score > 40, score=[*ctx.score, int(score)])


def preprocess_step7(ctx: Context):
    # STEP 7: normalize rotation
    if ctx.reject:
        return ctx

    imshow()

    norm = normalize(ctx.masked)

    structuring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    blur = normalize(cv2.bilateralFilter(norm, 32, 150, 150))

    _, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)

    keep = cv2.bitwise_and(thresh, ctx.mask_)
    morph_keep = morph_closes(keep, structuring)
    imshow(morph_keep)

    hull = get_hull(morph_keep, 2500)

    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    imshow_contours([np.intp(box)], (255, 0, 0))

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
    return replace(ctx, img=result, mask=None)


PREPROCESS_LIST = [
    preprocess_step1,
    preprocess_step2,
    preprocess_step3,
    preprocess_step4,
    preprocess_step5,
    preprocess_step6,
    reject_step6,
    preprocess_step7,
]


def preprocess(ctx: Context):
    for f in PREPROCESS_LIST:
        ctx = f(ctx)
    return ctx


if __name__ == "__main__":
    import logging

    logging.basicConfig()

    # TODO: cmd line
    IMG_PATHS = [
        Path("./data/dme/Response_9a88ec32f1c2ffe761e3824aab93c7b8_R_004.jpg"),
        Path("./data/dme/Response_45a2a8a4ea7ac0423248f93046523dc0_R_001.jpg"),
        Path("./data/dme/Response_f3bc8bc543ac8f076627c71ea8efaf35_L_012.jpg"),
        Path("./data/dme/Non response_0b9cf0f939daa80b6f7c1457ad104140_L_012.jpg"),
        Path("./data/dme/Response_bff83e94923f91e8af14db5714a937d9_L_001.jpg"),
        Path("./data/dme/Response_81f40dc0d3e0ec390a4960070fa8e609_L_000.jpg"),
        Path("./data/dme/Response_9c26b6082b53bbfcb23fee3cf17cafcf_L_007.jpg"),
        Path("./data/dme/Non response_8c5f0feeb9f34ff1f7b5325bd7855b60_L_007.jpg"),
    ]

    img = cv2.imread(str(IMG_PATHS[4]))
    result = preprocess(
        Context(
            cv2.cvtColor(img[:496, 496:, :], cv2.COLOR_BGR2GRAY),
        )
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
