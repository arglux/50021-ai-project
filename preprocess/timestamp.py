from headers import *

import datetime
import pandas as pd

daykey = {
	"Sun":0,
	"Mon":1,
	"Tue":2,
	"Wed":3,
	"Thu":4,
	"Fri":5,
	"Sat":6
	}

monthkey = {
	"Jan":1,
	"Feb":2,
	"Mar":3,
	"Apr":4,
	"May":5,
	"Jun":6,
	"Jul":7,
	"Aug":8,
	"Sep":9,
	"Oct":10,
	"Nov":11,
	"Dec":12
	}

def extract_timestamp_features(data, col_name):
	day = []
	month = []
	sec = [] # number of seconds from 1st Jan 2019

	for i in data[col_name]:
		t = i.split(" ")
		day.append(daykey[t[0]])
		month.append(monthkey[t[1]])
		date = datetime.datetime.strptime(i, '%a %b %d %H:%M:%S +0000 %Y')
		sec.append(str((date - datetime.datetime(2019, 1, 1)).total_seconds())[:-2])

	# simply append the return values  to DataFrame > df['col_name'] = ...
	return pd.Series(day), pd.Series(month), pd.Series(sec)

if __name__ == '__main__':
	# Timestamp: Format ( "EEE MMM dd HH:mm:ss Z yyyy" ). ISOString => integer (e.g. 23517957).

	data = pd.read_csv("../data/TweetsCOV19_052020.tsv.gz", compression='gzip', names=headers, sep='\t', quotechar='"')
	print("OK")

	# test
	day, month, sec = extract_timestamp_features(data[:10], 'Timestamp')
	print(day)
	print(month)
	print(sec)



