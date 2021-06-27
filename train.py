from model import Net
from save import save

import torch
import torch.nn as nn

import pandas as pd

def train(x, y, model, loss_function, optimizer, device, epochs=1000+1):
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
	inp_size = 49
	out_size = 1
	lr_rate = 0.001
	device = "cuda"

	model = Net(inp_size, out_size).to(device)
	loss_function = nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

	# assuming the data is downloaded and saved in the same dir -> ./content
	train_data = pd.read_csv('./data/herremans_hit_1030training.csv')
	print(f"Data loaded. Train data shape: {train_data.shape}")

	x = torch.FloatTensor(train_data.loc[:, train_data.columns != 'Topclass1030'].values).to(device)
	y = torch.FloatTensor(train_data['Topclass1030']).to(device)

	train(x, y, model, loss_function, optimizer, device)
	save(model, 'dummy-model')
