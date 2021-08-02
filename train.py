from model import Net, LinReg2
from save import save

import torch
import torch.nn as nn

import pandas as pd

class RMSLELoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.mse = nn.MSELoss()

	def forward(self, pred, actual):
		return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

def train(train_data, model, num_epochs):
	for e in range(num_epochs):
		batch_losses = []
		for i, (Xb, yb) in enumerate(train_data, model):
			_X = Variable(Xb).cuda().float()
			_y = Variable(yb).cuda().float()
			preds = model(_X)
			loss = criterion(preds, _y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			batch_losses.append(loss.item())

		mbl = np.mean(np.sqrt(batch_losses)).round(3)

		if e % 5 == 0: print("Epoch [{}/{}], Batch loss: {}".format(e, num_epochs, mbl))

def train_dummy(x, y, model, loss_function, optimizer, device, epochs=1000+1):
	for epoch in range(epochs):
		features = x.to(device)
		target = y.to(device)

		optimizer.zero_grad()

		prediction = model(features)
		loss = loss_function(prediction, target.view(-1, 1)).to(device)
		loss.backward()
		optimizer.step()

		if epoch % 50 == 0: # print every 50 epochs
		    print (f"Epoch: {epoch}, Loss: {loss}")

if __name__ == "__main__":
	input_size = 65
	hidden_size = 10
	output_size = 1
	learning_rate = 0.01
	num_epochs = 50

	model = LinReg2(input_size, hidden_size, output_size)
	criterion = RMSLELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

	if torch.cuda.is_available():
		model.cuda()
		print(model, criterion, optimizer)

	# model = Net(inp_size, out_size).to(device)
	# loss_function = nn.BCELoss()
	# optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

	# # assuming the data is downloaded and saved in the same dir -> ./content
	# train_data = pd.read_csv('./data/herremans_hit_1030training.csv')
	# print(f"Data loaded. Train data shape: {train_data.shape}")

	# x = torch.FloatTensor(train_data.loc[:, train_data.columns != 'Topclass1030'].values).to(device)
	# y = torch.FloatTensor(train_data['Topclass1030']).to(device)

	# train(x, y, model, loss_function, optimizer, device)
	# save(model, 'dummy-model')
