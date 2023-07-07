import cv2
from dataclasses import dataclass, field
from typing import Callable
import inspect


@dataclass
class Context:
    img: cv2.Mat
    reject: bool = field(default=False)
    score: list[int] = field(default_factory=list)
    dbg: Callable | bool = field(default=False)


def imshow(winname, mat):
    """
    magic imshow function that only show image in "debug mode"
    """
    for frameinfo in inspect.stack():
        if (
            (ctx := frameinfo.frame.f_locals.get("ctx", None))
            and isinstance(ctx, Context)
            and ctx.dbg
        ):
            if inspect.isfunction(ctx.dbg) and frameinfo.function != ctx.dbg.__name__:
                continue
            cv2.imshow(winname, mat)
            return


def imshow_contours(winname, contours, color):
    """
    magic imshow function that only show contours in "debug mode"
    """
    for frameinfo in inspect.stack():
        if (
            (ctx := frameinfo.frame.f_locals.get("ctx", None))
            and isinstance(ctx, Context)
            and ctx.dbg
        ):
            if inspect.isfunction(ctx.dbg) and frameinfo.function != ctx.dbg.__name__:
                continue
            cv2.imshow(
                winname,
                cv2.drawContours(
                    cv2.cvtColor(ctx.img, cv2.COLOR_GRAY2BGR),
                    contours,
                    -1,
                    color,
                ),
            )
            return
