import torch
import re
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
import numpy as np
import pandas as pd

TEXT_VECTOR_SIZE = 31


class Commit:
	_LABEL_TO_ID = {"p": 0, "c": 1, "a": 2}
	_PROJECT_TO_ID = {"reactivex-rxjava": 0, "hbase": 1, "elasticsearch": 2,
					  "intellij-community": 3, "hadoop": 4, "drools": 5,
					  "kotlin": 6, "restlet-framework-java": 7,
					  "orientdb": 8, "camel": 9,
					  "spring-framework": 10}

	_text_vectorizer = TfidfVectorizer(max_features=TEXT_VECTOR_SIZE)
	_all_messages_list = []

	def __init__(self, commit_id, project, message, label, other_features):
		self.commit_id = commit_id
		self.project = project
		self.message = message
		self.label = label
		self.other_features = list(other_features)
		self._all_messages_list.append(preprocess_text(message))

	@staticmethod
	def prepare_text_vectorizer():
		Commit._text_vectorizer.fit_transform(Commit._all_messages_list)

	def get_message_vector(self):
		return list(self._text_vectorizer.transform([preprocess_text(self.message)]).toarray()[0])

	def get_project(self):
		return self._PROJECT_TO_ID[self.project.lower()]

	def get_all_features_list(self):
		return self.get_message_vector() + [self.get_project()] + self.other_features

	def get_all_features_tensor(self):
		return torch.tensor(self.get_all_features_list())

	def get_label(self):
		return self._LABEL_TO_ID[self.label]

	def get_labels_tensor(self):
		labels_list = [-1, -1, -1]
		labels_list[self.get_label()] = 1
		labels_tensor = torch.tensor(labels_list)
		return labels_tensor

	def get_labels_list(self):
		labels_list = [-1, -1, -1]
		labels_list[self.get_label()] = 1
		return labels_list


def preprocess_text(text):
    text = re.sub("<[^>]*>", "", text)
    symbols = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = (re.sub("[\W]+", " ", text.lower()) + " ".join(symbols).replace("-", ""))
    return text
