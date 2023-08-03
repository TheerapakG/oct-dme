from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np
import pandas as pd
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


@dataclass
class Data:
    value: Path
    group: str
    side: str
    label: str

    @classmethod
    def from_path(cls, p: Path):
        name = p.stem.split("_")
        return cls(p, group=name[2], side=name[3], label=name[1])


class Splitter:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def split(self, ratio: list[float]):
        count = (
            self.data.groupby(["group", "label"])["value"].count().unstack().fillna(0)
        )
        count_sum = count.sum()
        ratio_sum = sum(ratio)
        target = pd.DataFrame([round(count_sum * r / ratio_sum) for r in ratio[:-1]])
        result = [pd.DataFrame() for _ in range(len(ratio))]
        for group, row in count.iloc[np.random.permutation(len(count))].iterrows():
            for i in range(len(ratio) - 1):
                if all(target.iloc[i] - row > 0):
                    target.iloc[i] -= row
                    result[i] = pd.concat(
                        [result[i], self.data[self.data["group"] == group]]
                    )
                    break
            else:
                result[-1] = pd.concat(
                    [result[-1], self.data[self.data["group"] == group]]
                )
        return result

    def split_best(self, ratio: list[float], n: int):
        labels = self.data["label"].unique()

        best_mse = math.inf
        best = [pd.DataFrame() for _ in range(len(ratio))]

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("splitting", total=n)
            for i in range(n):
                progress.update(task, completed=i)
                current = self.split(ratio)
                current_count = [
                    np.array(
                        [df.groupby(["label"])["value"].count()[l] for l in labels]
                    )
                    for df in current
                ]
                current_ratio = [count / sum(count) for count in current_count]

                if any([any(r == 0) for r in current_ratio]):
                    continue

                current_mse = sum(
                    ((c_a_ratio - c_b_ratio) ** 2).mean()
                    for c_a_ratio, c_b_ratio in combinations(current_ratio, 2)
                )
                if current_mse < best_mse:
                    best_mse = current_mse
                    best = current
                progress.update(task, completed=i + 1)

        return best


class Upsampler:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def upsample(self):
        # TODO: introduce randomness
        count = (
            self.data.groupby(["group", "label"])["value"].count().unstack().fillna(0)
        )
        count_sum = count.sum()
        target_sum = max(count_sum)

        result = pd.DataFrame()

        for label in count_sum.index:
            label_count = count[count[label] > 0].sort_values(label)[label]
            if count_sum[label] == target_sum:
                append = self.data[self.data["label"] == label].copy()
                append["duplicate"] = 0
                result = pd.concat([result, append])
                continue

            took_sum = 0
            for i in range(1, len(label_count)):
                took_sum += label_count.iloc[i - 1]
                fill = target_sum - (count_sum[label] - took_sum)
                if fill / i <= label_count.iloc[i]:
                    n, r = divmod(int(fill), i)
                    target = pd.concat(
                        [
                            pd.Series(n + 1, index=label_count.index[:r]),
                            pd.Series(n, index=label_count.index[r:i]),
                            label_count.iloc[i:],
                        ]
                    )
                    break
            else:
                n, r = divmod(int(target_sum), len(label_count))
                target = pd.concat(
                    [
                        pd.Series(n + 1, index=label_count.index[:r]),
                        pd.Series(n, index=label_count.index[r:]),
                    ]
                )

            for group, group_label_count in target.items():
                group_label_data = self.data[
                    (self.data["group"] == group) & (self.data["label"] == label)
                ]
                append = pd.DataFrame()

                n, r = divmod(int(group_label_count), len(group_label_data))
                for i in range(n):
                    append = pd.concat(
                        [
                            append,
                            pd.concat(
                                [
                                    group_label_data,
                                    pd.Series(
                                        i,
                                        index=group_label_data.index,
                                        name="duplicate",
                                    ),
                                ],
                                axis=1,
                            ),
                        ]
                    )
                append = pd.concat(
                    [
                        append,
                        pd.concat(
                            [
                                group_label_data.iloc[:r],
                                pd.Series(
                                    n,
                                    index=group_label_data.index[:r],
                                    name="duplicate",
                                ),
                            ],
                            axis=1,
                        ),
                    ]
                )
                result = pd.concat([result, append])

        print(
            result.groupby(["group", "label"])["value"]
            .count()
            .unstack()
            .fillna(0)
            .sum()
        )

        return result


def split(data: list[Data]):
    train, valid, test = Splitter(data).split_best([0.8, 0.1, 0.1], 64)

    print("train")
    print(train.groupby(["label"])["value"].count())
    print("valid")
    print(valid.groupby(["label"])["value"].count())
    print("test")
    print(test.groupby(["label"])["value"].count())

    list_groups_set = [
        (label, set(groups))
        for label, groups in [
            ("train", train["group"].unique()),
            ("valid", valid["group"].unique()),
            ("test", test["group"].unique()),
        ]
    ]
    print(
        "intersection",
        [
            ((label_a, label_b), len(s_a & s_b))
            for (label_a, s_a), (label_b, s_b) in combinations(list_groups_set, 2)
        ],
    )

    return train, valid, test


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-w",
        "--weight",
        action="store_true",
        help="output weight files instead of performing class balancing",
    )
    args = parser.parse_args()

    DATA_PATH = Path("./data/filter_dme_dbg/prep_accept")
    SPLIT_PATH = Path("./data/split_dme_dbg/")

    data = pd.DataFrame([Data.from_path(p) for p in DATA_PATH.iterdir()])
    train, valid, test = split(data)

    dss = {
        "train": train,
        "valid": valid,
        "test": test,
    }

    SPLIT_PATH.mkdir(parents=True)
    for s, ds in dss.items():
        if args.weight:
            count = (
                ds.groupby(["group", "side", "label"])["value"]
                .count()
                .unstack()
                .fillna(0)
            )
            count.to_csv(SPLIT_PATH / f"{s}_weight.csv")
            for label in ds["label"].unique():
                for _, row in ds[ds["label"] == label].iterrows():
                    (SPLIT_PATH / s / label / row["group"] / row["side"]).mkdir(
                        parents=True, exist_ok=True
                    )
                    shutil.copy2(
                        row["value"],
                        SPLIT_PATH / s / label / row["group"] / row["side"],
                    )
        else:
            up = Upsampler(ds).upsample()
            for label in up["label"].unique():
                (SPLIT_PATH / s / label).mkdir(parents=True, exist_ok=True)
                for _, row in up[up["label"] == label].iterrows():
                    p = row["value"]
                    shutil.copy2(
                        p,
                        SPLIT_PATH
                        / s
                        / row["label"]
                        / f"{p.stem}_{row['duplicate']}.{p.suffix}",
                    )


if __name__ == "__main__":
    main()
