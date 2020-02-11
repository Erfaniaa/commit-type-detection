# Commit Type Detection
Classify Git commits with deep learning

# Introduction

According to this (https://arxiv.org/pdf/1711.05340.pdf) paper:

We suppose that there are 3 main classification categories for software project maintenance activities:

**Corrective**: fixing faults (functional and non-functional)

**Perfective**: improving the system and its design

**Adaptive**: introducing new features into the system

In this work, we seek to design a commit classification model capable of providing high accuracy to detect these three types of commits.

The used dataset can be found here: https://zenodo.org/record/835534

# Method

In the mentioned paper, three algorithms have been used and compared. Among J48, GBM, and RF algorithms, RF had a better performance.

Instead of using these algorithms, we implemented a **deep learning** approach. Here you can see the implemented neural network architecture (copied from network.py file):

```python
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
```

As you can read, a fully-connected neural network has been implemented in **PyTorch** deep learning framework.

In our dataset, each commit has a message, project name, and 68 other features. By applying **tf-idf** algorithm on the commit messages, we may convert each commit data to a vector with size 100. So, the input of this network is a vector with a size equal to 100.

Like the paper method, our models were trained using 85% of the dataset, while the remaining 15% was used as a test set.

# Result

A confusion matrix will be shown after training. You can compare this data to the 8th table of the mentioned paper. As you can see, our method has reached **74.5% accuracy** in this case.


```
Predict  a        c        p        
Actual
a        17       4        10       

c        5        74       6        

p        3        16       38       




Overall Statistics:

Kappa                                                      0.57912
NIR                                                        0.49133
Overall Accuracy                                           0.74566
P-Value [Accuracy > NIR]                                   0.0

Class Statistics:

Classes                                                    Adaptive    Corrective  Perfective
ACC(Accuracy)                                              0.87283     0.82081     0.79769
ERR(Error rate)                                            0.12717     0.17919     0.20231
FN(False negative/miss/type 2 error)                       14          11          19
FP(False positive/type 1 error/false alarm)                8           20          16
FPR(Fall-out or false positive rate)                       0.05634     0.22727     0.13793
PPV(Precision or positive predictive value)                0.68        0.78723     0.7037
TN(True negative/correct rejection)                        134         68          100
TNR(Specificity or true negative rate)                     0.94366     0.77273     0.86207
TP(True positive/hit)                                      17          74          38
TPR(Sensitivity, recall, hit rate, or true positive rate)  0.54839     0.87059     0.66667
```
# Usage

Use Python version 3.

First of all, install the required Python packages:

```bash
pip install requirements.txt
```

And then run the Python program:

```
python main.py
```

