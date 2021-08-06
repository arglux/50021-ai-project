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

if __name__ == "__main__":
	input_size = 65
	hidden_size = 10
	output_size = 1
	learningRate = 0.01

	model = LinReg2(input_size, hidden_size, output_size)

	if torch.cuda.is_available():
		model.cuda()
		print(model)

