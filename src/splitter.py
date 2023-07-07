from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations
import math
import numbers
import numpy as np
from pathlib import Path
import shutil
from sklearn.utils import check_random_state, indexable
from typing import Generic, Iterable, TypeVar

_T = TypeVar("_T")


@dataclass
class Data(Generic[_T]):
    value: _T
    group: str
    label: str


# Modified from sklearn
def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


class BaseShuffleSplit(ABC):
    def __init__(self, n_splits=10, split_size=[0.8, 0.1, 0.1], *, random_state=None):
        self.n_splits = n_splits
        self.split_size = split_size
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)  # type: ignore
        for split in self._iter_indices(X, y, groups):
            yield split

    @abstractmethod
    def _iter_indices(self, X, y=None, groups=None) -> Iterable[tuple[np.ndarray, ...]]:
        pass


class ShuffleSplit(BaseShuffleSplit):
    def __init__(self, n_splits=10, split_size=[0.8, 0.1, 0.1], *, random_state=None):
        super().__init__(
            n_splits=n_splits,
            split_size=split_size,
            random_state=random_state,
        )

    def _iter_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        n_split_acc = np.add.accumulate(self.split_size)
        n_split_size = [
            0,
            *np.floor((n_split_acc * n_samples) / n_split_acc[-1]).astype(np.int32),
        ]

        rng = check_random_state(self.random_state)
        for _ in range(self.n_splits):
            permutation = rng.permutation(n_samples)
            yield tuple(
                permutation[s:e] for s, e in zip(n_split_size[:-1], n_split_size[1:])
            )


class GroupShuffleSplit(ShuffleSplit):
    def __init__(self, n_splits=5, split_size=[0.8, 0.1, 0.1], *, random_state=None):
        super().__init__(
            n_splits=n_splits,
            split_size=split_size,
            random_state=random_state,
        )

    def _iter_indices(self, X, y, groups):
        classes, group_indices = np.unique(groups, return_inverse=True)
        for group_labels in super()._iter_indices(X=classes):
            yield tuple(
                np.flatnonzero(np.isin(group_indices, group_label))
                for group_label in group_labels
            )

    def split(self, X, y=None, groups=None):
        return super().split(X, y, groups)


def split(data: list[Data]):
    keys = {d.label for d in data}

    x = np.array([d.value for d in data])
    y = np.array([d.label for d in data])
    groups = np.array([d.group for d in data])

    train_test_valid_mse = math.inf
    train_x, test_x, valid_x = np.array([]), np.array([]), np.array([])
    train_y, test_y, valid_y = np.array([]), np.array([]), np.array([])
    train_groups, test_groups, valid_groups = np.array([]), np.array([]), np.array([])
    for train, test, valid in GroupShuffleSplit(20, [0.8, 0.1, 0.1]).split(
        x, y, groups=groups
    ):
        c_train_x, c_test_x, c_valid_x = x[train], x[test], x[valid]
        c_train_y, c_test_y, c_valid_y = y[train], y[test], y[valid]
        c_train_groups, c_test_groups, c_valid_groups = (
            groups[train],
            groups[test],
            groups[valid],
        )
        c_train_ratio = np.array(
            [len(c_train_y[c_train_y == key]) for key in keys]
        ) / len(c_train_y)
        c_test_ratio = np.array([len(c_test_y[c_test_y == key]) for key in keys]) / len(
            c_test_y
        )
        c_valid_ratio = np.array(
            [len(c_valid_y[c_valid_y == key]) for key in keys]
        ) / len(c_valid_y)
        if any(c_train_ratio == 0) or any(c_test_ratio == 0) or any(c_valid_ratio == 0):
            continue
        c_mse = sum(
            ((c_a_ratio - c_b_ratio) ** 2).mean()
            for c_a_ratio, c_b_ratio in combinations(
                [c_train_ratio, c_test_ratio, c_valid_ratio], 2
            )
        )
        if c_mse < train_test_valid_mse:
            train_test_valid_mse = c_mse
            train_x, test_x, valid_x = c_train_x, c_test_x, c_valid_x
            train_y, test_y, valid_y = c_train_y, c_test_y, c_valid_y
            train_groups, test_groups, valid_groups = (
                c_train_groups,
                c_test_groups,
                c_valid_groups,
            )

    print("train", [(key, len(train_y[train_y == key])) for key in keys])
    print("test", [(key, len(test_y[test_y == key])) for key in keys])
    print("valid", [(key, len(valid_y[valid_y == key])) for key in keys])

    list_groups_set = [
        (label, set(groups))
        for label, groups in [
            ("train", train_groups),
            ("test", test_groups),
            ("valid", valid_groups),
        ]
    ]
    print(
        "intersection",
        [
            ((label_a, label_b), len(s_a & s_b))
            for (label_a, s_a), (label_b, s_b) in combinations(list_groups_set, 2)
        ],
    )

    return train_x, test_x, valid_x


if __name__ == "__main__":
    DATA_PATH = Path("./data/filter_dme_dbg/prep_accept")
    SPLIT_PATH = Path("./data/split_dme_dbg/")

    data = []
    for p in DATA_PATH.iterdir():
        name = p.stem.split("_")
        data.append(Data(p, group=name[2], label=name[1]))

    train_x, test_x, valid_x = split(data)

    SPLIT_PATH.mkdir(parents=True)
    for label, paths in [("train", train_x), ("test", test_x), ("valid", valid_x)]:
        (SPLIT_PATH / label).mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy2(p, SPLIT_PATH / label)
