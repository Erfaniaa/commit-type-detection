from torch import nn, optim
from torch.nn import functional as F

LEARNING_RATE = 1e-3

class Network(nn.Module):
    def __init__(self, input_size=68, output_size=3):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4 = nn.Linear(10, output_size)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = torch.sigmoid(self.fc4(h3))
        return h4


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def loss(desired_output, network_output):
    mse = nn.MSELoss(network_output, desired_output, size_average=True, reduction="mean")
    return mse
