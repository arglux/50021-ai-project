from model import LinReg2
from gensim.models import Word2Vec

from preprocess.headers import *
from preprocess.sentiment import *
from preprocess.timestamp import *
from preprocess.utils import *
from dataset import CreateDataset

import math
import torch
import numpy as np
import pandas as pd

from torch.autograd import Variable
from torch.utils.data import DataLoader

def load_model(model_path, inp_size, hidden_size, out_size, device='cpu'):
	trained_model = LinReg2(inp_size, hidden_size, out_size).to(device) # reinitialize model
	trained_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
	trained_model.eval()
	return trained_model

def predict(model, dataloader, convert=True):
	# print(model, dataloader)
	result = []
	for _X, _y in dataloader:
		_X = Variable(_X).float()
		_y = Variable(_y).float()

		pred = model(_X)

		# print(pred.item())
		if convert: return (math.floor(10**_y-1), max(0, math.floor(10**pred.item()-1)))
		else: return (_y, pred)

def coerce_datatype(inp, mention_embeddings, hashtag_embeddings, lookup_user=None):
	"""
	Convert app's input to df with following data:
	headers =	[
			"Tweet Id",
			"Username",
			"Timestamp",
			"#Followers",
			"#Friends",
			"#Retweets",
			"#Favorites",
			"#Entities",
			"Sentiment",
			"Mentions",
			"Hashtags",
			"URLs",
		]

	TO:
	(in the form of dataloader of batch 32)
	clean_headers = ['Hashtag Emb0', 'Hashtag Emb1', 'Hashtag Emb2', 'Hashtag Emb3',
       'Hashtag Emb4', 'Hashtag Emb5', 'Hashtag Emb6', 'Hashtag Emb7',
       'Hashtag Emb8', 'Hashtag Emb9', 'Hashtag Emb10', 'Hashtag Emb11',
       'Hashtag Emb12', 'Hashtag Emb13', 'Hashtag Emb14', 'Hashtag Emb15',
       'Hashtag Emb16', 'Hashtag Emb17', 'Hashtag Emb18', 'Hashtag Emb19',
       'Hashtag Emb20', 'Hashtag Emb21', 'Hashtag Emb22', 'Hashtag Emb23',
       'Hashtag Emb24', 'Mention Emb0', 'Mention Emb1', 'Mention Emb2',
       'Mention Emb3', 'Mention Emb4', 'Mention Emb5', 'Mention Emb6',
       'Mention Emb7', 'Mention Emb8', 'Mention Emb9', 'Mention Emb10',
       'Mention Emb11', 'Mention Emb12', 'Mention Emb13', 'Mention Emb14',
       'Mention Emb15', 'Mention Emb16', 'Mention Emb17', 'Mention Emb18',
       'Mention Emb19', 'Mention Emb20', 'Mention Emb21', 'Mention Emb22',
       'Mention Emb23', 'Mention Emb24', 'scaled_Positive', 'scaled_Negative',
       'scaled_Sentiment Disparity', 'log_#Followers', 'log_#Friends',
       'log_No. of Entities', 'log_#Favorites', 'log_Time Int',
       'ohe_Day of Week_1', 'ohe_Day of Week_2', 'ohe_Day of Week_3',
       'ohe_Day of Week_4', 'ohe_Day of Week_5', 'ohe_Day of Week_6',
       'ohe_Day of Week_7', '#Followers_min', '#Friends_min', '#Retweets_min',
       '#Favorites_min', '#Followers_max', '#Friends_max', '#Retweets_max',
       '#Favorites_max', '#Followers_mean', '#Friends_mean', '#Retweets_mean',
       '#Favorites_mean', 'label'
		]
	"""
	# print('raw input:', inp)

	out = pd.DataFrame(inp)

	# begin in-place transformation of different columns
	day, month, sec = extract_timestamp_features(out, 'Timestamp')
	out['Day of Week'] = day
	# out['Month'] = month # dropped because we only have 8 months of data in one of the set
	out['Time Int'] = sec

	positive, negative, disparity = extract_sentiment_features(out, 'Sentiment')
	out['Positive'] = positive
	out['Negative'] = negative
	out['Sentiment Disparity'] = disparity
	out['No. of Entities'] = out['#Entities']

	out = vectorize_and_append(out, 'Hashtags', mention_embeddings, 'Hashtag')
	out = vectorize_and_append(out, 'Mentions', mention_embeddings, 'Mention')

	for x in ['Tweet Id', 'Timestamp', '#Entities', 'Sentiment', 'Mentions', 'Hashtags', 'URLs']:
		del out[x]

	out = scaledtransform(out, [("Positive", 5), ("Negative", -5), ("Sentiment Disparity", 10)])
	out = logtransform(out, ["#Retweets", "#Followers", "#Friends", "No. of Entities", "#Favorites", "Time Int"])
	out = single_ohetransform(out, ["Day of Week"])
	out = unpackcol(out, ["ohe_Day of Week"])

	dataset = CreateDataset(out, protocol="df", lookup_path=lookup_user)
	dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
	# print(out.columns, len(out.columns))

	return dataloader

if __name__ == '__main__':
	inp = {'Tweet Id': ['1259153416082747392'], 'Username': ['1bf6a0f57c9f08faf3aa804f83539df6'], 'Timestamp': ['Sat May 09 16:09:02 +0000 2020'], '#Followers': [716], '#Friends': [72], '#Retweets': ['13'], '#Favorites': [8], '#Entities': [3], 'Sentiment': ['1 -1'], 'Mentions': ['torghost'], 'Hashtags': ['Tornetwork. Python3. CyberSec bugbounty linux pentest tools infosec Covid_19 COVID19 StayAtHome StayHomeStaySafe'], 'URLs': [0]} # example of app's parsed input

	hashtag_embeddings = Word2Vec.load('./data/hashtag_embeddings')
	mention_embeddings = Word2Vec.load('./data/mention_embeddings')

	lookup_user = pd.read_csv("./data/TweetsCOV19_052020.tsv.gz", compression='gzip', names=headers, sep='\t', quotechar='"')
	model_inp = coerce_datatype(inp, mention_embeddings, hashtag_embeddings, lookup_user=lookup_user)

	model_path = './models/77InpLinReg-0608-1746'
	out_size = 1
	hidden_size = 32
	inp_size = 77
	model = load_model(model_path, inp_size, hidden_size, out_size)

	model_out = predict(model, model_inp)
	print(f"With user lookup table: {model_out}") # (true, pred)
