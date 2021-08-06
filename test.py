from model import LinReg2
from train import RMSLELoss
from predict import load_model
from dataset import CreateDataset
from datetime import datetime

from torch.autograd import Variable
from torch.utils.data import DataLoader

import math
import torch
import numpy as np
import pandas as pd

def test(model, dataloader, criterion, optimizer):
	accs = []
	losses = []
	for _X, _y in dataloader:
		_X = Variable(_X).cuda().float()
		_y = Variable(_y).cuda().float()

		test_preds = model(_X)
		test_loss = criterion(test_preds, _y)

		losses.append(test_loss.item())
		accs.append(magnitude_diff(math.floor(10**_y-1), max(0, math.floor(10**test_preds.item()-1))))

	return losses, accs

def magnitude_diff(y, pred):
	return (np.log10(abs(y - pred) + 1))

if __name__ == '__main__':
	model_path = './models/77InpLinReg-0608-1746'
	out_size = 1
	hidden_size = 32
	inp_size = 77
	learning_rate = 0.001

	model = load_model(model_path, inp_size, hidden_size, out_size, device='cuda')
	criterion = RMSLELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

	dataset = CreateDataset("./data/77test.pkl", protocol="pickle")
	test_dl = DataLoader(dataset, batch_size=1, shuffle=False)

	print("Running test...")
	losses, accs = test(model, test_dl, criterion, optimizer)

	now = datetime.now()
	timestamp = now.strftime("%d%m-%H%M")
	textfile = open(f"./data/test_loss_{timestamp}.txt", "w")
	for loss in losses: textfile.write(str(loss) + "\n")

	textfile = open(f"./data/accuracy_{timestamp}.txt", "w")
	for acc in accs: textfile.write(str(acc) + "\n")

