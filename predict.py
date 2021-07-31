from model import Net

import torch
import pandas as pd

def load_model(model_path, inp_size, out_size, device='cpu'):
	trained_model = Net(inp_size, out_size).to(device) # reinitialize model
	trained_model.load_state_dict(torch.load(model_path))
	trained_model.eval()
	return trained_model

def predict(model, input):
	print(model, input)

if __name__ == '__main__':
	print("OK")
