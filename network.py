import torch
from torch import nn, optim
from torch.nn import functional as F

NETWORK_INPUT_SIZE = 100
NETWORK_OUTPUT_SIZE = 3


class Network(nn.Module):
	def __init__(self, input_size=NETWORK_INPUT_SIZE, output_size=NETWORK_OUTPUT_SIZE):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(input_size, 80)
		self.fc2 = nn.Linear(80, 60)
		self.dropout1 = nn.Dropout(0.01)
		self.fc3 = nn.Linear(60, 40)
		self.fc4 = nn.Linear(40, 20)
		self.fc5 = nn.Linear(20, output_size)

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout1(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.dropout1(x)
		x = self.fc3(x)
		x = F.relu(x)
		x = self.dropout1(x)
		x = self.fc4(x)
		x = F.relu(x)
		x = self.dropout1(x)
		x = self.fc5(x)
		x = torch.tanh(x)
		return x


def normal_init(m, mean, std):
	if isinstance(m, nn.Linear):
		m.weight.data.normal_(mean, std)
		m.bias.data.zero_()


def loss(desired_output, network_output):
	mse = torch.mean(((network_output - desired_output) ** 2))
	return mse
