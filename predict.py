from model import Net
from gensim.models import Word2Vec

from preprocess.headers import *
from preprocess.sentiment import *
from preprocess.timestamp import *
# from preprocess.utils import vectorize_using_embedding

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
	# hashtag_embeddings = Word2Vec.load('./data/hashtag_embeddings')
	# mention_embeddings = Word2Vec.load('./data/mention_embeddings')
	print('raw input:', inp)

	values = {}

	values['#Followers'] = inp['#Followers']
	values['#Friends'] = inp['#Friends']
	values['#Retweets'] = [0] # TODO: replace with actual values from test set
	values['#Favorites'] = inp['#Favorites']

	positive, negative, disparity = extract_sentiment_features(pd.DataFrame(inp), 'Sentiment')
	values['Positive'] = positive
	values['Negative'] = negative
	values['Disparity'] = disparity

	values['No. of Entities'] = [0]

	day, month, sec = extract_timestamp_features(pd.DataFrame(inp), 'Timestamp')
	values['Day of Week'] = day
	values['Month'] = month
	values['Time Int'] = sec

	# values['Mention Embedding'] = vectorize_using_embedding(inp['Mentions'], mention_embeddings)
	# values['Hashtag Embedding'] = vectorize_using_embedding(inp['Hashtags'], hashtag_embeddings)

	print('values:', values)
	out = pd.DataFrame(values) # must pass in list for each key's value

	return out

if __name__ == '__main__':
	inp = {'#Followers': [854], '#Friends': [217], '#Favorites': [843], 'Sentiment': ['-5 5'], 'Timestamp': ['Sat Jul 31 14:25:21 +0000 2021'], 'Mentions': ['realDonaldTrump'], 'Hashtags': ['COVID19'], 'URLs': ['www.google.com']}

	model_inp = coerce_datatype(inp)
	print(model_inp)
