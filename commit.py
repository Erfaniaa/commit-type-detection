import torch
from torch import tensor


class Commit:
    
    def __init__(self, commit_id, project, message, label, features):
        self.commit_id = commit_id
        self.project = project
        self.message = message
        self.label = label
        self.features = features

    def to_tensor(self):
        return torch.tensor(self.features) / 1000.0
