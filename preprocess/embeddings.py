import logging
import numpy as np

from utils import *
from headers import *
from gensim.models import Word2Vec

class Corpus(object):
	def __init__(self, df, column, null='null;', delimiter=' '):
		self.df = df[df[column] != null].dropna()
		self.column = column
		self.delimiter = delimiter

	def __iter__(self): # a memory-friendly iterator
		for idx, row in self.df.iterrows():
			sentence = row[self.column].split(self.delimiter)
			yield sentence

def train_embeddings(data, column, save_path, embedding_dim=25, min_count=200, workers=4):
	non_null_column_data, _ = get_non_null_column_sorted_by_retweet(data, column)
	sentences = Corpus(non_null_column_data, column)
	model = Word2Vec(sentences, vector_size=embedding_dim, min_count=min_count, workers=workers)
	model.save(save_path)
	return model

if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	fpath = "../data/TweetsCOV19_052020.tsv.gz"
	data = pd.read_csv(fpath, compression='gzip', names=headers, sep='\t', quotechar='"')

	column = 'Hashtags'
	hashtags_embeddings_path = '../data/hashtag_embeddings'
	hashtags_embeddings = train_embeddings(data, column, hashtags_embeddings_path)

	column = 'Mentions'
	mentions_embeddings_path = '../data/mention_embeddings'
	mentions_embeddings = train_embeddings(data, column, mentions_embeddings_path)
