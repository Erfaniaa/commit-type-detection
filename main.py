import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
from commit import Commit


DATASET_FILENAME = "dataset.csv"
DATASET_FILE_DELIMITER = "#"


def read_commits_data_from_dataset():
	global commits
	commits = []
	csv_dataset = pd.read_csv(DATASET_FILENAME, delimiter=DATASET_FILE_DELIMITER)
	rows = len(csv_dataset)
	cols = len(csv_dataset.iloc[0])
	for i in range(rows):
		data = csv_dataset.iloc[i]
		commit = Commit(data["commitId"], data["project"], data["comment"], data["label"], data[4:])
		commits.append(commit)


def initialize():
	global device
	global model
	global optimizer
	global train_loader
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = Network().to(device)
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


def train(epochs=10):
	for i in range(epochs):
		for j in range(commits):
			network_output = model()
			single_loss = loss(network_output, train_outputs)

			if i % 25 == 1:
				print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

			optimizer.zero_grad()
			single_loss.backward()
			optimizer.step()

		print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


if __name__ == "__main__":
	initialize()
    read_commits_data_from_dataset()
