import torch
import torch.nn as nn

class LinReg2(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(LinReg2, self).__init__()
		self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
		self.relu_h1 = nn.ReLU()
		self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)

	def forward(self, X):
		out = self.relu_h1(self.fc1(X))
		out = self.fc2(out)
		return out


class Net(nn.Module):
	def __init__(self, input_size, num_classes):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input_size, 32)
		self.fc2 = nn.Linear(32, 8)
		self.fc3 = nn.Linear(8, num_classes)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(p=0.75)

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		x = self.relu(x)
		x = self.sigmoid(x)
		return x

if __name__ == "__main__":
	input_size = 65
	hidden_size = 10
	output_size = 1
	learningRate = 0.01

	model = LinReg2(input_size, hidden_size, output_size)

	if torch.cuda.is_available():
		model.cuda()
		print(model)

