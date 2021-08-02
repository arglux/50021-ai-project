import numpy as np
import pandas as pd

def get_non_null_column_sorted_by_retweet(data, column, complete=False, null='null;'):
	sorted_by_retweets = data.sort_values(by='#Retweets', ascending=False)
	non_null_sorted_by_retweets = sorted_by_retweets[(sorted_by_retweets[column] != null)].dropna()

	# TODO: lower alphabet? COVID19 -> covid19

	if complete: return (non_null_sorted_by_retweets, len(non_null_sorted_by_retweets))
	else: return (non_null_sorted_by_retweets[['#Followers', column]], len(non_null_sorted_by_retweets))

def count(data, column, delimiter=" "):
	counts = {}
	total = 0

	for idx, row in data.iterrows():
		values = row[column].split(delimiter)

		for value in values:
			value = value.replace(",", "") # remove those ending with a comma
			if value not in counts: counts[value] = 1
			else: counts[value] += 1
			total += 1

	return counts, total

def sort_counts_by_frequency(counts, total, cutoff=0.2, min_frequency=200):
	counts_sorted_by_frequency = sorted(counts.items(), key=lambda item: item[1], reverse=True)
	most_popular_list = []

	cum_sum = 0
	for (value, frequency) in counts_sorted_by_frequency:
		if frequency <= min_frequency: break # skip all tags that just occur once

		percentage = frequency / total
		cum_sum += percentage
		if (cum_sum < cutoff): most_popular_list.append([value, frequency])
		else: break

	return most_popular_list

def vectorize_using_embedding(values, embedding, size=25):
	print(values)
	zero = np.zeros(size)

	for val in values:
		if val in embedding.wv.index_to_key:
			# print(embedding.wv[val])
			zero += embedding.wv[val]
		else: continue

	return zero


if __name__ == '__main__':
	print("OK")



