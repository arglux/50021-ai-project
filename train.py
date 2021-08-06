from dataset import CreateDataset
from model import LinReg2
from save import save
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader

class RMSLELoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.mse = nn.MSELoss()

	def forward(self, pred, actual):
		return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

def train(model, dataloader, criterion, optimizer, num_epochs=50):
	mbls = []
	for e in range(num_epochs+1):
		batch_losses = []
		for i, (Xb, yb) in enumerate(dataloader):
			_X = Variable(Xb).cuda().float()
			_y = Variable(yb).cuda().float()
			preds = model(_X)
			loss = criterion(preds, _y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			batch_losses.append(loss.item())

		mbl = np.mean(np.sqrt(batch_losses)).round(3)

		if e % 1 == 0:
			print("Epoch [{}/{}], Batch loss: {}".format(e, num_epochs, mbl))
			mbls.append(mbl)

	return mbls

if __name__ == "__main__":
	input_size = 77
	hidden_size = 32
	output_size = 1
	learning_rate = 0.001
	num_epochs = 15

	model = LinReg2(input_size, hidden_size, output_size)
	criterion = RMSLELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

	if torch.cuda.is_available():
		model.cuda()
		print(model, criterion, optimizer)

	dataset = CreateDataset("./data/77train.pkl", protocol="pickle")
	train_dl = DataLoader(dataset, batch_size=32, shuffle=True)

	losses = train(model, train_dl, criterion, optimizer, num_epochs)
	save(model, '77InpLinReg')

	now = datetime.now()
	timestamp = now.strftime("%d%m-%H%M")
	textfile = open(f"./data/train_loss_{timestamp}.txt", "w")
	for loss in losses: textfile.write(str(loss) + "\n")

