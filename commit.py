import torch
from torch import tensor


class Commit:
    
    def __init__(self, commit_id, project, message, label, features):
        self.commit_id = commit_id
        self.project = project
        self.message = message
        self.label = label
        self.features = features

    def get_tensor(self):
        return torch.tensor(self.features) / 1000.0

    def get_label(self):
    	return {"p": 0, "c": 1, "a": 2}[self.label]

    def get_labels_vector(self):
    	labels_vector = [0, 0, 0]
    	labels_vector[self.get_label()] = 1
    	return labels_vector
