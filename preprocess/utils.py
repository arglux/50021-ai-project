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

def vectorize_and_append(dataframe, target_col, embedding, out_col, size=25):
	vocabs = embedding.wv.index_to_key
	ls = []

	for i in dataframe[target_col]:
		if i in vocabs and i != "null;":
			ls.append(embedding.wv[i])
		else:
			ls.append(np.zeros((size,), dtype=int))

	for i in range(size):
		dataframe[f"{out_col} Emb{str(i)}"] = pd.Series([x[i] for x in ls])

	return dataframe

def scaledtransform(dataframe, cols):
	for col,m in cols:
		dataframe["scaled_" + col ] = dataframe[col].apply(lambda x: x/m)
		dataframe = dataframe.drop(col,1)
	return dataframe

def logtransform(dataframe, cols):
	for col in cols:
		dataframe["log_" + col ] = dataframe[col].apply(lambda x: np.log10(int(x)+1))
		dataframe = dataframe.drop(col,1)
	return dataframe

def ohetransform(dataframe, cols):
	for col in cols:
		one_hot = pd.get_dummies(dataframe[col])
		print(one_hot)
		dataframe["ohe_" + col ] = one_hot.values.tolist()
		dataframe = dataframe.drop(col,1)
	return dataframe

def single_ohetransform(dataframe, cols):
	for col in cols:
		one_hot = [0, 0, 0, 0, 0, 0, 0]
		one_hot[dataframe[col].values[0]] = 1
		dataframe["ohe_" + col ] = [one_hot]
		dataframe = dataframe.drop(col,1)
	return dataframe

def unpackcol(dataframe,cols):
	for col in cols:
		unpacked = pd.DataFrame(dataframe[col].tolist(), columns=[f'{col}_{idx + 1}' for idx in range(len(dataframe[col].values[0]))], index= dataframe.index)
		dataframe = dataframe.drop(col,axis=1)
		dataframe = pd.concat([dataframe, unpacked], axis=1, join='inner')
	return dataframe

if __name__ == '__main__':
	print("OK")



