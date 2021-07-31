from model import Net
from preprocess.headers import *
from preprocess.sentiment import *
from preprocess.timestamp import *

import torch
import pandas as pd

def load_model(model_path, inp_size, out_size, device='cpu'):
	trained_model = Net(inp_size, out_size).to(device) # reinitialize model
	trained_model.load_state_dict(torch.load(model_path))
	trained_model.eval()
	return trained_model

def predict(model, inp):
	print(model, inp)

def coerce_datatype(inp, spread_vector=False):
	"""
	Convert app's input to df with following data:
	clean_headers = [
		'#Followers',
		'#Friends',
		'#Retweets',
		'#Favorites',
		'Positive',
		'Negative',
		'Sentiment Disparity',
		'No. of Entities',
		'Day of Week',
		'Month',
		'Time Int',
		'Mention Embedding',
		'Hashtag Embedding'
		]
	"""
	print(inp)

	values = inp[:2] #Followers, #Friends
	values.append(0) #Retweets
	values.append(inp[2]) #Favorites


	print(values)
	print(inp)

	inp_dict = dict(zip(clean_headers, [[val] for val in values])) # must pass in list for each key's value
	out = pd.DataFrame(inp_dict)
	return out

if __name__ == '__main__':
	inp = [825, 773, 33, '-2 - 1', 'Sat Jul 31 13:47:52 +0000 2021', 'realDonaldTrump', 'COVID19', 'www.google.com']

	model_inp = coerce_datatype(inp)
	print(model_inp)
