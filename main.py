import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
from commit import Commit
import random
import network
from pycm import ConfusionMatrix


CSV_FILENAME = "dataset.csv"
CSV_FILE_DELIMITER = "#"
EPOCHS_COUNT = 100
BATCH_SIZE = 1
TRAINING_DATASET_SIZE_RATIO = 0.85
LEARNING_RATE = 0.00005


def get_batch_features_tensor(index, batch_size=BATCH_SIZE):
	return torch.tensor([training_dataset[i].get_all_features_list() for i in range(index, index + batch_size)])


def get_batch_labels_tensor(index, batch_size=BATCH_SIZE):
	return torch.tensor([training_dataset[i].get_labels_list() for i in range(index, index + batch_size)])


def prepare_training_and_validation_datasets(training_set_size_ratio=TRAINING_DATASET_SIZE_RATIO, batch_size=BATCH_SIZE):
	global training_dataset
	global validation_dataset
	global training_dataset_size
	global validation_dataset_size
	global batches_count
	random.shuffle(commits)
	training_dataset_size = (int(len(commits) * training_set_size_ratio) // batch_size) * batch_size
	training_dataset = commits[:training_dataset_size]
	validation_dataset = commits[training_dataset_size:]
	validation_dataset_size = len(validation_dataset)
	batches_count = training_dataset_size // batch_size


def read_commits_data_from_csv(csv_filename=CSV_FILENAME, csv_file_delimiter=CSV_FILE_DELIMITER):
	global commits
	commits = []
	csv_contents = pd.read_csv(csv_filename, delimiter=csv_file_delimiter)
	rows = len(csv_contents)
	cols = len(csv_contents.iloc[0])
	for i in range(rows):
		data = csv_contents.iloc[i]
		commit = Commit(data["commitId"], data["project"], data["comment"], data["label"], data[4:])
		commits.append(commit)
	Commit.prepare_text_vectorizer()


def initialize_network():
	global device
	global model
	global optimizer
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = network.Network().to(device)
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	# for m in model._modules:
		# network.normal_init(model._modules[m], 0, 1)


def validate():
	global input_labels
	global predicted_labels
	input_labels = []
	predicted_labels = []
	model.eval()
	total_true_positives = 0
	with torch.no_grad():
		for j in range(validation_dataset_size):
			network_output = model(validation_dataset[j].get_all_features_tensor().to(device))
			network_output_index = int(network_output.argmax(dim=0, keepdim=False))
			if network_output_index == validation_dataset[j].get_label():
				total_true_positives += 1
			input_labels.append(validation_dataset[j].get_label())
			predicted_labels.append(network_output_index)
	accuracy = total_true_positives / validation_dataset_size
	print("Accuracy:", accuracy)


def print_confusion_matrix(input_labels, predicted_labels):
	cm = ConfusionMatrix(input_labels, predicted_labels)
	print(cm)


def train_all(epochs_count=EPOCHS_COUNT, batch_size=BATCH_SIZE):
	model.train()
	for i in range(epochs_count):
		training_loss = 0
		for j in range(batches_count):
			network_output = model(get_batch_features_tensor(j).to(device))
			optimizer.zero_grad()
			loss_value = network.loss(network_output, get_batch_labels_tensor(j))
			training_loss += loss_value.item()
			loss_value.backward()
			optimizer.step()
		training_loss /= batches_count
		print("Epoch number:", i + 1)
		print("Training loss:", training_loss)
		validate()
		print("---------------------")


if __name__ == "__main__":
	initialize_network()
	read_commits_data_from_csv()
	prepare_training_and_validation_datasets()
	train_all()
	print_confusion_matrix()
