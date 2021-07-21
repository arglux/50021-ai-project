from headers import *

import numpy as np
import pandas as pd

def get_non_null_hashtags_sorted_by_retweets(data, complete=False, null='null;'):
	sorted_by_retweets = data.sort_values(by='#Retweets', ascending=False)
	non_null_sorted_by_retweets = sorted_by_retweets[(sorted_by_retweets['Hashtags'] != null)].dropna()

	# TODO: lower alphabet? COVID19 -> covid19

	if complete: return (non_null_sorted_by_retweets, len(non_null_sorted_by_retweets))
	else: return (non_null_sorted_by_retweets[['#Followers', 'Hashtags']], len(non_null_sorted_by_retweets))

def count_hashtags(data, delimiter=" "):
	hashtag_counts = {}
	total = 0

	for idx, row in data.iterrows():
		values = row['Hashtags'].split(delimiter)

		for value in values:
			value = value.replace(",", "") # remove those ending with a comma
			if value not in hashtag_counts: hashtag_counts[value] = 1
			else: hashtag_counts[value] += 1
			total += 1

	return hashtag_counts, total

def sort_hashtags_by_frequency(hashtag_counts, cutoff=0.2, min_frequency=500):
	hashtags_sorted_by_frequency = sorted(hashtag_counts.items(), key=lambda item: item[1], reverse=True)
	most_popular_hastags = []

	cum_sum = 0
	for (hashtag, frequency) in hashtags_sorted_by_frequency:
		if frequency <= min_frequency: break # skip all tags that just occur once

		percentage = frequency / total
		cum_sum += percentage
		if (cum_sum < cutoff): most_popular_hastags.append([hashtag, frequency])
		else: break

	return most_popular_hastags

if __name__ == '__main__':
	fpath = "../data/TweetsCOV19_052020.tsv.gz"
	data = pd.read_csv(fpath, compression='gzip', names=headers, sep='\t', quotechar='"')

	clean_data, _ = get_non_null_hashtags_sorted_by_retweets(data)
	hashtag_counts, total = count_hashtags(clean_data)
	most_popular_hastags = sort_hashtags_by_frequency(hashtag_counts)

	array = np.array(most_popular_hastags)
	np.savetxt("../data/popular_hashtags.csv", array, delimiter=", ", fmt='%s', encoding='utf8')
	print(array)
