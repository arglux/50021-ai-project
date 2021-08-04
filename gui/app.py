import sys
import random

sys.path.append("../") # to access predict.py

from predict import *
from preprocess.headers import headers
from datetime import datetime

from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg

from main_window import Ui_Form
from PIL import Image

class Main(qtw.QWidget, Ui_Form):
	"""
	handles user interaction, loads data and updates GUI
	"""
	def __init__(self):
		"""
		initializes and sets up GUI widgets and its connections
		"""
		super().__init__()
		self.setupUi(self)
		self.setWindowTitle("Retweet Prediction App")

		# state values
		self.prediction_value_label = "-"
		self.true_value_label = "-"
		self.values = []

		print("Loading... This may take a while (~3 mins) depending on test set size...") # line below
		self.data = pd.read_csv("../data/TweetsCOV19_052020.tsv.gz", compression='gzip', names=headers, sep='\t', quotechar='"')

		# attach button to function
		self.randomizeButton.clicked.connect(self.randomize)
		self.predictButton.clicked.connect(self.predict)

		# setup
		self._update_values(self.prediction_value_label, self.predictionValueLabel)
		self._update_values(self.true_value_label, self.trueValueLabel)

		self.headers = [
			"Tweet Id",
			"Username",
			"Timestamp",
			"#Followers",
			"#Friends",
			"#Retweets",
			"#Favorites",
			"#Entities",
			"Sentiment",
			"Mentions",
			"Hashtags",
			"URLs",
		]

		# load model
		model_path = '../models/dummy-model-2706-2356'
		inp_size = 49
		out_size = 1
		device = 'cpu'
		self.model = load_model(model_path, inp_size, out_size)

	def randomize(self):
		index = random.randint(0, len(self.data.index))
		data_point = self.data.iloc[index]
		self.tweet_id = str(data_point['Tweet Id'])
		self.username = str(data_point['Username'])

		# now = str(datetime.now().strftime('%a %b %d %H:%M:%S +0000 %Y')) # EEE MMM dd HH:mm:ss Z yyyy

		self.numOfFollowersEdit.setText( str(data_point['#Followers']) )
		self.numOfFriendsEdit.setText( str(data_point['#Friends']) )
		self.numOfFavoritesEdit.setText( str(data_point['#Favorites']) )

		self.sentimentEdit.setText( str(data_point['Sentiment']) )
		self.datetimeEdit.setText( str(data_point['Timestamp']) )

		self.mentionsEdit.setText( str(data_point['Mentions']) )
		self.hashtagsEdit.setText( str(data_point['Hashtags']) )

		self.entitiesCountEdit.setText( str(len(data_point['Entities'].split(' '))) )

		self.trueValueIndex.setText(f'Data referenced. Index: {index}. Tweet Id: {self.tweet_id}.')
		self.true_value_label = str(data_point['#Retweets'])
		self._update_values(self.true_value_label, self.trueValueLabel)

	def predict(self):
		# read values
		self.values = self._read_values()

		# coerce datatypes
		self.input = coerce_datatype( dict(zip(self.headers, self.values)) )

		# feed into model
		predict(self.model, self.input)

		# show result
		self.prediction_value_label = str(random.randint(0, 1000))
		self._update_values(self.prediction_value_label, self.predictionValueLabel)

	def _read_values(self):
		# all values are read as list to standardize (easier to turn into DataFrame also)
		numOfFollowers = int(self.numOfFollowersEdit.text())
		numOfFriends = int(self.numOfFriendsEdit.text())
		numOfFavorites = int(self.numOfFavoritesEdit.text())
		sentiment = self.sentimentEdit.text()
		datetime = self.datetimeEdit.text()
		mentions = self.mentionsEdit.text()
		hashtags = self.hashtagsEdit.text()
		entitiesCount = int(self.entitiesCountEdit.text())

		return([
		       	[self.tweet_id],
		       	[self.username],
		       	[datetime],
		      	[numOfFollowers],
		      	[numOfFriends],
		      	[self.true_value_label], # #Retweets
		      	[numOfFavorites],
		      	[entitiesCount],
		      	[sentiment],
		      	[mentions],
		      	[hashtags],
		      	[0], # URLs
		      ])

	def _update_values(self, string, label):
		label.setText(string)
		print(f"{label} set to {string}")

	def center_window(self):
		"""
		centers the fixed main window size according to user screen size
		"""
		screen = qtw.QDesktopWidget().screenGeometry()
		main_window = self.geometry()
		x = (screen.width() - main_window.width()) / 2

		# pulls the window up slightly (arbitrary)
		y = (screen.height() - main_window.height()) / 2 - 50
		self.setFixedSize(main_window.width(), main_window.height())
		self.move(x, y)

	def clear_layout(self, layout):
		"""
		clear all widget within a layout
		"""
		while layout.count() > 0:
			exist = layout.takeAt(0)
			if not exist: continue
			else: exist.widget().deleteLater()

if __name__ == "__main__":
	app = qtw.QApplication([])
	main = Main()
	main.center_window()
	main.show()
	sys.exit(app.exec_())
