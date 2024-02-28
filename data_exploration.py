import os
import pandas as pd

data_folder = "datasets"

train_articles_folder = os.path.join(data_folder, "train-articles")
dev_articles_folder = os.path.join(data_folder, "dev-articles")

train_articles_flc_tc_labels = os.path.join(data_folder, "train-task-flc-tc.labels")
dev_articles_flc_tc_labels = os.path.join(data_folder, "dev-task-flc-tc.labels")

task = "tc"

def load_data():
    pdf_train_labels = pd.read_csv(train_articles_flc_tc_labels, sep="\t", header=None, names=["article", "label", "start_i", "end_i"])
    
    pdf_train_labels['article'] = pdf_train_labels['article'].apply(lambda x: int(str(x)[-4:]))




if __name__ == "__main__":
    load_data()