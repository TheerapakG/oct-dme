from collections.abc import Mapping
import cv2
from dataclasses import dataclass, field, replace
from functools import wraps
import inspect
import logging
import numpy as np
from typing import Callable


@dataclass
class Context:
    img: cv2.Mat
    mask: cv2.Mat | None = field(default=None)
    reject: bool = field(default=False)
    score: list[int] = field(default_factory=list)
    dbg: Callable | bool | None = field(default=None)

    @property
    def mask_(self):
        if self.mask is None:
            self.mask = (np.ones(self.img.shape) * 255).astype(self.img.dtype)
        return self.mask

    @property
    def masked(self):
        return cv2.bitwise_and(self.img, self.img, mask=self.mask_)

    def overlay_mask(self, color):
        img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        color_img = np.zeros(img.shape, img.dtype)
        color_img[:, :] = color
        mask_img = cv2.bitwise_and(color_img, color_img, mask=self.mask_)
        return cv2.addWeighted(mask_img, 1, img, 1, 0)

    @staticmethod
    def get_current():
        return next(
            (
                _ctx
                for frameinfo in inspect.stack()
                if (_ctx := frameinfo.frame.f_locals.get("ctx", None))
                and isinstance(_ctx, Context)
            ),
            None,
        )


def is_dbg_ctx(ctx: Context | None):
    return bool(
        ctx
        and ctx.dbg
        and (
            isinstance(ctx.dbg, bool)
            or ctx.dbg.__name__ in [frameinfo.function for frameinfo in inspect.stack()]
        )
    )


def is_dbg_current_ctx():
    return is_dbg_ctx(Context.get_current())


class ContextLogger(logging.LoggerAdapter):
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)

    def log(
        self,
        level: int,
        msg: object,
        *args: object,
        exc_info=None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **kwargs,
    ):
        prev_level = self.logger.level
        if is_dbg_current_ctx():
            self.setLevel(logging.DEBUG)

        super().log(
            level,
            msg,
            *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            **kwargs,
        )

        self.setLevel(prev_level)


log = ContextLogger(logging.getLogger(__name__))


def dbg(f):
    signature = inspect.signature(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        old_ctx = Context.get_current()

        if not old_ctx or old_ctx.dbg == False:
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            return f(*bound.args, **bound.kwargs)

        ctx = replace(old_ctx, dbg=f)

        bound = signature.bind_partial(*args, **kwargs)
        if "ctx" in signature.parameters:
            bound.arguments["ctx"] = ctx
        bound.apply_defaults()

        res = f(*bound.args, **bound.kwargs)
        if isinstance(res, Context):
            res = replace(ctx, dbg=old_ctx.dbg)

        return res

    return wrapper


def dbg_only(f):
    signature = inspect.signature(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        stack = inspect.stack()

        ctx = Context.get_current()

        if not is_dbg_ctx(ctx):
            return

        bound = signature.bind_partial(*args, **kwargs)
        if "arg_names" in signature.parameters:
            bound_args = {
                n: bound.arguments[n]
                for n in signature.parameters["arg_names"].default.keys()
                if n in bound.arguments
            }
            arg_names_dict = {
                n: [f_n for f_n, f_v in stack[1].frame.f_locals.items() if v is f_v]
                for n, v in bound_args.items()
            }
            bound.arguments["arg_names"] = {
                **signature.parameters["arg_names"].default,
                **{n: f_ns for n, f_ns in arg_names_dict.items() if f_ns},
                **bound.arguments.get("arg_names", {}),
            }

        if "caller" in signature.parameters and "caller" not in bound.arguments:
            bound.arguments["caller"] = stack[1].function

        if "ctx" in signature.parameters and "ctx" not in bound.arguments:
            bound.arguments["ctx"] = ctx

        bound.apply_defaults()

        return f(*bound.args, **bound.kwargs)

    return wrapper


@dbg_only
def imshow(
    img: cv2.Mat | None = None,
    *,
    name: str | None = None,
    arg_names: dict[str, list[str]] = {"img": ["unknown"]},
    caller="unknown",
    ctx: Context | None = None,
):
    """
    magic imshow function that only show image in "debug mode"
    """
    assert ctx
    cv2.imshow(
        name
        if name
        else ("original" if img is None else f"{caller}_{arg_names['img'][0]}"),
        img if img is not None else ctx.overlay_mask((0, 0, 63)),
    )


@dbg_only
def imshow_contours(
    contours,
    color,
    img: cv2.Mat | None = None,
    *,
    name: str | None = None,
    arg_names: dict[str, list[str]] = {"contours": ["unknown"]},
    caller="unknown",
    ctx: Context | None = None,
):
    """
    magic imshow function that only show contours in "debug mode"
    """
    assert ctx
    cv2.imshow(
        name if name else f"{caller}_{arg_names['contours'][0]}",
        cv2.drawContours(
            cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img is not None
            else ctx.overlay_mask((0, 0, 63)),
            contours,
            -1,
            color,
        ),
    )
