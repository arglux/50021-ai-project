import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import pickle5 as pickle

from preprocess.utils import find_user_stats

class CreateDataset(Dataset):
	def __init__(self, path, lookup_path=None, protocol="standard"):
		if protocol == "pickle":
			with open(path, "rb") as fh:
				data = pickle.load(fh)

			if lookup_path:
				with open(lookup_path, "rb") as fh:
					lookup = pickle.load(lookup_path)
				data = data.merge(lookup, how='inner', on='Username')

		elif protocol == "standard":
			data = pd.read_pickle(path)

			if lookup_path:
				lookup = pd.read_pickle(lookup_path)
				data = data.merge(lookup, how='inner', on='Username')

		elif protocol == "df":
			data = path

			if isinstance(lookup_path, pd.DataFrame):
				lookup = find_user_stats(lookup_path, protocol="df")
				data = data.merge(lookup, how='inner', on='Username')

		self.y = torch.from_numpy(data[['log_#Retweets']].values)
		# data['label'] = data['log_#Retweets'].apply(lambda x: float(0) if x==float(0) else float(1))
		data = data.drop(["log_#Retweets", "Username"], 1)
		self.X = torch.from_numpy(data.values)
		# print(data.columns, len(data.columns))
		# print(data)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

if __name__ == "__main__":
	dataset = CreateDataset("./data/77test.pkl", protocol="pickle")
	print(dataset)
