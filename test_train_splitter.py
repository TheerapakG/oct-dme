from collections import defaultdict
import cv2
from itertools import combinations
import math
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = Path("./data/dme/")
SPLIT_PATH = Path("./data/split_dme/")

data_bins: defaultdict[tuple[str, str], list[Path]] = defaultdict(list)
for p in DATA_PATH.iterdir():
    name = p.stem.split("_")
    data_bin = (name[0], name[1])
    img = cv2.imread(str(p))
    if img is None or img.size == 0:
        print(f"error loading {p}")
        continue
    data_bins[data_bin].append(p)
[(data_bin, len(paths)) for data_bin, paths in data_bins.items()]

keys = {data_bin[0] for data_bin in data_bins.keys()}

x = []
y = []
groups = []
for data_bin, paths in data_bins.items():
    x.extend(paths)
    y.extend([data_bin[0]] * len(paths))
    groups.extend([data_bin[1]] * len(paths))

x = np.array(x)
y = np.array(y)
groups = np.array(groups)

train_test_valid_mse = math.inf
train_x, test_valid_x = np.array([]), np.array([])
train_y, test_valid_y = np.array([]), np.array([])
train_groups, test_valid_groups = np.array([]), np.array([])
for train, test_valid in GroupShuffleSplit(10, train_size=0.6).split(x, y, groups=groups):
    c_train_x, c_test_valid_x = x[train], x[test_valid]
    c_train_y, c_test_valid_y = y[train], y[test_valid]
    c_train_groups, c_test_valid_groups = groups[train], groups[test_valid]
    c_train_ratio = np.array([len(c_train_y[c_train_y == key]) for key in keys]) / len(c_train_y)
    c_test_valid_ratio = np.array([len(c_test_valid_y[c_test_valid_y == key]) for key in keys]) / len(c_test_valid_y)
    if any(c_train_ratio == 0) or any(c_test_valid_ratio == 0):
        continue
    c_mse = ((c_train_ratio - c_test_valid_ratio) ** 2).mean()
    if c_mse < train_test_valid_mse:
        train_test_valid_mse = c_mse
        train_x, test_valid_x = c_train_x, c_test_valid_x
        train_y, test_valid_y = c_train_y, c_test_valid_y
        train_groups, test_valid_groups = c_train_groups, c_test_valid_groups

test_valid_mse = math.inf
test_x, valid_x = np.array([]), np.array([])
test_y, valid_y = np.array([]), np.array([])
test_groups, valid_groups = np.array([]), np.array([])
for test, valid in GroupShuffleSplit(10, train_size=0.5).split(test_valid_x, test_valid_y, groups=test_valid_groups):
    c_test_x, c_valid_x = test_valid_x[test], test_valid_x[valid]
    c_test_y, c_valid_y = test_valid_y[test], test_valid_y[valid]
    c_test_groups, c_valid_groups = test_valid_groups[test], test_valid_groups[valid]
    c_test_ratio = np.array([len(c_test_y[c_test_y == key]) for key in keys]) / len(c_test_y)
    c_valid_ratio = np.array([len(c_valid_y[c_valid_y == key]) for key in keys]) / len(c_valid_y)
    if any(c_test_ratio == 0) or any(c_valid_ratio == 0):
        continue
    c_mse = ((c_test_ratio - c_valid_ratio) ** 2).mean()
    if c_mse < test_valid_mse:
        test_valid_mse = c_mse
        test_x, valid_x = c_test_x, c_valid_x
        test_y, valid_y = c_test_y, c_valid_y
        test_groups, valid_groups = c_test_groups, c_valid_groups

print("train", [(key, len(train_y[train_y == key])) for key in keys])
print("test", [(key, len(test_y[test_y == key])) for key in keys])
print("valid", [(key, len(valid_y[valid_y == key])) for key in keys])

list_groups_set = [(label, set(groups)) for label, groups in [("train", train_groups), ("test", test_groups), ("valid", valid_groups)]]
print("intersection", [((label_a, label_b), len(s_a & s_b)) for (label_a, s_a), (label_b, s_b) in combinations(list_groups_set, 2)])

SPLIT_PATH.mkdir(parents=True)
for label, paths in [("train", train_x), ("test", test_x), ("valid", valid_x)]:
    (SPLIT_PATH / label).mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, SPLIT_PATH / label)
