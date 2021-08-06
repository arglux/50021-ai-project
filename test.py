from model import LinReg2
from train import RMSLELoss

from torch.utils.data import DataLoader

import torch
import pandas as pd

def train(model, dataloader):
	model.eval()
	for _X, _y in test_dl:
		test_batch_losses = []
		_X = Variable(_X).cuda().float()
		_y = Variable(_y).cuda().float()

		#apply model
		test_preds = model(_X)
		test_loss = criterion(test_preds, _y)

		for i in range(100):
			print(math.floor(10**_y[i]-1),math.floor(10**test_preds[i].item()-1))
		break

	# print(model, dataloader)
	result = []
	for _X, _y in dataloader:
		_X = Variable(_X).float()
		_y = Variable(_y).float()

		pred = model(_X)

		# print(pred.item())
		if convert: return (math.floor(10**_y-1), math.floor(10**pred.item()-1))
		else: return (_y, pred)

if __name__ == '__main__':
	print("OK")
	# inp_size = 49
	# out_size = 1
	# device = 'cpu'

	# trained_model = Net(inp_size, out_size).to(device) # reinitialize model
	# trained_model.load_state_dict(torch.load('./models/dummy-model-2706-2356'))
	# trained_model.eval()

	# test_data = pd.read_csv('./data/herremans_hit_1030test.csv')
	# print(f"Data loaded. Train data shape: {test_data.shape}")

	# x = torch.FloatTensor(test_data.loc[:, test_data.columns != 'Topclass1030'].values).to(device)
	# y = torch.FloatTensor(test_data['Topclass1030']).to(device)

	# sensitivity_evaluation(trained_model, x, y)
