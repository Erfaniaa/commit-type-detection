import pandas as pd
from commit import Commit

if __name__ == "__main__":
    dataset = pd.read_csv("dataset.csv", delimiter="#")
    rows = len(dataset)
    cols = len(dataset.iloc[0])
    for i in range(rows):
        data = dataset.iloc[i]
        commit = Commit(data["commitId"], data["project"], data["comment"], data["label"], data[4:])
        print(commit.to_tensor())
    