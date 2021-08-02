from model import Net
from gensim.models import Word2Vec

from preprocess.headers import *
from preprocess.sentiment import *
from preprocess.timestamp import *
from preprocess.utils import vectorize_using_embedding

import torch
import numpy as np
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
		'#Retweets',
		'Positive',
		'Negative',
		'Sentiment Disparity',
		'Log_#Followers',
		'Log_#Friends',
		'Log_No. of Entities',
		'Log_#Favorites',
		'Log_Time Int',
		'Mention Embedding_1',
		'Mention Embedding_2',
		'Mention Embedding_3',
		'Mention Embedding_4',
		'Mention Embedding_5',
		'Mention Embedding_6',
		'Mention Embedding_7',
		'Mention Embedding_8',
		'Mention Embedding_9',
		'Mention Embedding_10',
		'Mention Embedding_11',
		'Mention Embedding_12',
		'Mention Embedding_13',
		'Mention Embedding_14',
		'Mention Embedding_15',
		'Mention Embedding_16',
		'Mention Embedding_17',
		'Mention Embedding_18',
		'Mention Embedding_19',
		'Mention Embedding_20',
		'Mention Embedding_21',
		'Mention Embedding_22',
		'Mention Embedding_23',
		'Mention Embedding_24',
		'Mention Embedding_25',
		'Hashtag Embedding_1',
		'Hashtag Embedding_2',
		'Hashtag Embedding_3',
		'Hashtag Embedding_4',
		'Hashtag Embedding_5',
		'Hashtag Embedding_6',
		'Hashtag Embedding_7',
		'Hashtag Embedding_8',
		'Hashtag Embedding_9',
		'Hashtag Embedding_10',
		'Hashtag Embedding_11',
		'Hashtag Embedding_12',
		'Hashtag Embedding_13',
		'Hashtag Embedding_14',
		'Hashtag Embedding_15',
		'Hashtag Embedding_16',
		'Hashtag Embedding_17',
		'Hashtag Embedding_18',
		'Hashtag Embedding_19',
		'Hashtag Embedding_20',
		'Hashtag Embedding_21',
		'Hashtag Embedding_22',
		'Hashtag Embedding_23',
		'Hashtag Embedding_24',
		'Hashtag Embedding_25',
		'1-Hot_Day of Week_1',
		'1-Hot_Day of Week_2',
		'1-Hot_Day of Week_3',
		'1-Hot_Day of Week_4',
		'1-Hot_Day of Week_5',
		'1-Hot_Day of Week_6',
		'1-Hot_Day of Week_7',
		]
	"""
	print('raw input:', inp)

	hashtag_embeddings = Word2Vec.load('./data/hashtag_embeddings')
	mention_embeddings = Word2Vec.load('./data/mention_embeddings')

	inp['Mention Embedding'] = vectorize_using_embedding(list(inp['Mentions']), mention_embeddings)
	inp['Hashtag Embedding'] = vectorize_using_embedding(list(inp['Hashtags']), hashtag_embeddings)
	inp.pop('Mentions')
	inp.pop('Hashtags')

	print(inp)
	for k, v in inp.items():
		print(k, len(v))

	out = pd.DataFrame(inp) # must pass in list for each key's value

	return out

def scaledtransform(dataframe, cols):
	for col,m in cols:
		dataframe["scaled_" + col ] = dataframe[col].apply(lambda x: x/m)
		dataframe = dataframe.drop(col,1)
	return dataframe

def logtransform(dataframe,cols):
	for col in cols:
		dataframe["log_" + col ] = dataframe[col].apply(lambda x: np.log10(int(x)+1))
		dataframe = dataframe.drop(col,1)
	return dataframe

def ohetransform(dataframe, cols):
	for col in cols:
		one_hot = pd.get_dummies(dataframe[col])
		dataframe["ohe_" + col ] = one_hot.values.tolist()
		dataframe = dataframe.drop(col,1)
	return dataframe

def unpackcol(dataframe,cols):
	for col in cols:
		unpacked = pd.DataFrame(df[col].tolist(), columns=[f'{col}_{idx + 1}' for idx in range(len(df[col].values[0]))], index= dataframe.index)
		dataframe = dataframe.drop(col,axis=1)
		dataframe = pd.concat([dataframe, unpacked], axis=1, join='inner')
	return dataframe

if __name__ == '__main__':
	inp = {'#Followers': [16662], '#Friends': [9595], '#Favorites': [5], 'Sentiment': ['1 -1'], 'Timestamp': ['Sat May 09 09:44:13 +0000 2020'], 'Mentions': ['EmpressEphiya', 'donsummerone', 'CRUCIALQUALITY1', 'ansah_apagya', 'SaddickAdams', 'realDonaldTrump', 'Dela_fishbone', 'newtonlartey6', 'AngelfmAccra', 'Angel961Fm', 'QuamiCopper', 'Nana_Wiser', 'MawukoDoe', 'jeffreyamoah', 'fathyskinny', 'KwameBrooklyn4', 'gazanation_1'], 'Hashtags': ['angelsports', 'angelsports', 'angelsportsaccra', 'angelsports', 'angelsportsaccra', 'angelsports', 'angelsportsaccra', 'angelsports', 'angelsportsaccra', 'angelsports', 'angelsportsaccra', 'angelsports', 'angelsportsaccra', 'angelsports'], 'No. of Entities': [1]}

	model_inp = coerce_datatype(inp)
	print(model_inp)
