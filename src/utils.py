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


def dbg_only(*, arg_names: list[str] = []):
    def decorator(f):
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
                    and ctx.dbg.__name__
                    not in [frameinfo.function for frameinfo in stack]
                )
            ):
                return

            if arg_names:
                bound = signature.bind_partial(*args, **kwargs)
                bound_args = {
                    n: bound.arguments[n] for n in arg_names if n in bound.arguments
                }
                arg_names_dict = {
                    n: [f_n for f_n, f_v in stack[1].frame.f_locals.items() if v is f_v]
                    for n, v in bound_args.items()
                }
                kwargs["arg_names"] = {
                    n: f_ns for n, f_ns in arg_names_dict.items() if f_ns
                }

            if "caller" in signature.parameters and "caller" not in kwargs:
                kwargs["caller"] = stack[1].function

            if "ctx" in signature.parameters and "ctx" not in kwargs:
                kwargs["ctx"] = ctx

            return f(*args, **kwargs)

        return wrapper

    return decorator


@dbg_only(arg_names=["mat"])
def imshow(
    mat,
    *,
    name: str | None = None,
    arg_names: dict[str, list[str]] = {},
    caller="unknown",
):
    """
    magic imshow function that only show image in "debug mode"
    """
    cv2.imshow(
        name if name else f"{caller}_{arg_names.get('mat', ['unknown'])[0]}", mat
    )


@dbg_only(arg_names=["contours"])
def imshow_contours(
    contours,
    color,
    *,
    name: str | None = None,
    arg_names: dict[str, str] = {},
    caller="unknown",
    ctx: Context | None = None,
):
    """
    magic imshow function that only show contours in "debug mode"
    """
    assert ctx
    cv2.imshow(
        name if name else f"{caller}_{arg_names.get('contours', ['unknown'])[0]}",
        cv2.drawContours(
            cv2.cvtColor(ctx.img, cv2.COLOR_GRAY2BGR),
            contours,
            -1,
            color,
        ),
    )
