from utils import *
from headers import *

import numpy as np
import pandas as pd

def get_popular_hastags_by_frequency(fpath, save_path, preview=True):
	column = 'Hashtags'
	data = pd.read_csv(fpath, compression='gzip', names=headers, sep='\t', quotechar='"')

	non_null_column_data, _ = get_non_null_column_sorted_by_retweet(data, column)
	hashtag_counts, total = count(non_null_column_data, column)
	most_popular_hastags = sort_counts_by_frequency(hashtag_counts, total)

	array = np.array(most_popular_hastags)
	np.savetxt(save_path, array, delimiter=", ", fmt='%s', encoding='utf8')
	if preview: print(array)
	return array

if __name__ == '__main__':
	fpath = "../data/TweetsCOV19_052020.tsv.gz"
	save_path = "../data/popular_hashtags.csv"
	most_popular_hastags = get_popular_hastags_by_frequency(fpath, save_path)


