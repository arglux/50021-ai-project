from model import Net

import torch
import pandas as pd

# load trained model
inp_size = 49
out_size = 1
device = 'cpu'

trained_model = Net(inp_size, out_size).to(device) # reinitialize model
trained_model.load_state_dict(torch.load('./models/dummy-model-2706-2356'))
trained_model.eval()

test_data = pd.read_csv('./data/herremans_hit_1030test.csv')
print(f"Data loaded. Train data shape: {test_data.shape}")

x = torch.FloatTensor(test_data.loc[:, test_data.columns != 'Topclass1030'].values).to(device)
y = torch.FloatTensor(test_data['Topclass1030']).to(device)

def predict(input):
	print("OK")


if __name__ == '__main__':
	inp = "OK"
	predict(inp)
