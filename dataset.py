import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd

class Dataset(Dataset):
	def __init__(self, path, limit=None):
		data = pd.read_pickle(path)
		data = logtransform(data,["#Retweets"])
		self.y = torch.from_numpy(data[['log_#Retweets']].values)
		data=data.drop("log_#Retweets", 1)
		self.X = torch.from_numpy(data.values)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

if __name__ == "__main__":
	dataset = Dataset("./data/mil.pkl")
	print(dataset)
