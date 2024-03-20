import csv
from sklearn.metrics import f1_score

y_true = []
with open("datasets/dev-task-flc-tc.labels", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        y_true += [row[1]]

y_pred = []
with open("output/output", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        y_pred += [row[1]]


print(f1_score(y_true, y_pred, average="weighted"))
# current bert model has an F1 score of 0.11 :((
