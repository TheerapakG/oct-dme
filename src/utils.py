import cv2
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable
import inspect


@dataclass
class Context:
    img: cv2.Mat
    reject: bool = field(default=False)
    score: list[int] = field(default_factory=list)
    dbg: Callable | bool = field(default=False)


def dbg_only(f):
    signature = inspect.signature(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        ctx: Context | None = None

        stack = inspect.stack()

        for frameinfo in stack:
            if (
                ctx is None
                and (_ctx := frameinfo.frame.f_locals.get("ctx", None))
                and isinstance(_ctx, Context)
            ):
                ctx = _ctx

        if (
            not ctx
            or not ctx.dbg
            or (
                inspect.isfunction(ctx.dbg)
                and ctx.dbg.__name__ not in [frameinfo.function for frameinfo in stack]
            )
        ):
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
        name if name else f"{caller}_{arg_names['img'][0]}",
        img if img is not None else ctx.img,
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
            cv2.cvtColor(img if img is not None else ctx.img, cv2.COLOR_GRAY2BGR),
            contours,
            -1,
            color,
        ),
    )
