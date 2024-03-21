import csv
import os

from sklearn.metrics import f1_score


def get_f1(folder):
    y_true = []
    with open("datasets/dev-task-flc-tc.labels", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            y_true += [row[1]]

    y_pred = []
    path = os.path.join("_output", folder, "output")
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            y_pred += [row[1]]

    return f1_score(y_true, y_pred, average="weighted")


folders = [
    "_epoch_1",
    "_epoch_3",
    "_epoch_5",
    "seq_len_64",
    "seq_len_128",
    "seq_len_256",
    "seq_len_512",
]

for folder in folders:
    print("folder", folder, "has f1", get_f1(folder))
