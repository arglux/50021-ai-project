from headers import *

import pandas as pd

def extract_sentiment_features(data, col_name):
	positive = pd.Series([ int(x.split(' ')[0]) for x in data[col_name] ])
	negative = pd.Series([ int(x.split(' ')[1]) for x in data[col_name] ])
	disparity = pd.Series([ int(x.split(' ')[0]) - int(x.split(' ')[1]) for x in data[col_name] ])

	# simply append the return values  to DataFrame > df['col_name'] = ...
	return positive, negative, disparity

if __name__ == '__main__':
	# Timestamp: Format ( "EEE MMM dd HH:mm:ss Z yyyy" ). ISOString => integer (e.g. 23517957).

	data = pd.read_csv("../data/TweetsCOV19_052020.tsv.gz", compression='gzip', names=headers, sep='\t', quotechar='"')
	print("OK")

	# test
	positive, negative, disparity = extract_sentiment_features(data[:10], 'Sentiment')
	print(positive)
	print(negative)
	print(disparity)



