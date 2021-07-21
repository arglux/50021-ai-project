from utils import *
from headers import *

import numpy as np
import pandas as pd

def get_popular_mentions_by_frequency(fpath, save_path, preview=True):
	column = 'Mentions'
	data = pd.read_csv(fpath, compression='gzip', names=headers, sep='\t', quotechar='"')

	non_null_column_data, _ = get_non_null_column_sorted_by_retweet(data, column)
	mention_counts, total = count(non_null_column_data, column)
	mentions_by_frequency = sort_counts_by_frequency(mention_counts, total, cutoff=0.1)
		# NOTE: at 0.2 -> 720 unique mentions, so reduce it for now to 0.1

	array = np.array(mentions_by_frequency)
	np.savetxt(save_path, array, delimiter=", ", fmt='%s', encoding='utf8')
	if preview: print(array)
	return array

if __name__ == '__main__':
	fpath = "../data/TweetsCOV19_052020.tsv.gz"
	save_path = "../data/popular_mentions.csv"
	most_popular_mentions = get_popular_mentions_by_frequency(fpath, save_path)


