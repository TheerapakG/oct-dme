import cv2
from pathlib import Path
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)
import shutil

from .preprocess import Context, preprocess


if __name__ == "__main__":
    DATA_PATH = Path("./data/dme/")
    RESULT_PATH = Path("./data/filter_dme_dbg/")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[path]}"),
    ) as progress:
        task = progress.add_task(
            "preprocessing", total=len([None for _ in DATA_PATH.iterdir()]), path=None
        )
        for i, path in enumerate(DATA_PATH.iterdir()):
            progress.update(task, completed=i, path=path)
            img = cv2.imread(str(path))
            if img is None or img.size == 0:
                print("error", f'"{path}"')
                continue

            ctx = preprocess(
                Context(cv2.cvtColor(img[:496, 496:, :], cv2.COLOR_BGR2GRAY))
            )

            res_dir = RESULT_PATH / ("reject" if ctx.reject else "accept")
            res_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                path, res_dir / f"{'_'.join(str(s) for s in ctx.score)}_{path.name}"
            )

            prep_dir = RESULT_PATH / ("prep_reject" if ctx.reject else "prep_accept")
            prep_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(prep_dir / f"{'_'.join(str(s) for s in ctx.score)}_{path.name}"),
                ctx.img,
            )
