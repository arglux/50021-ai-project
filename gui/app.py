import sys
import random

sys.path.append("../") # to access predict.py

from predict import *
from preprocess.headers import headers

from datetime import datetime
from gensim.models import Word2Vec

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
		self.value_selected_randomly_from_dataset = False
		self.values = []

		print("Loading... This may take a while (~3 mins) depending on test set size...") # line below
		self.data = pd.read_csv("../data/TweetsCOV19_052020.tsv.gz", compression='gzip', names=headers, sep='\t', quotechar='"')
		self.hashtag_embeddings = Word2Vec.load('../data/hashtag_embeddings')
		self.mention_embeddings = Word2Vec.load('../data/mention_embeddings')

		# attach button to function
		self.randomizeButton.clicked.connect(self.randomize)
		self.predictButton.clicked.connect(self.predict)

		# attach listeners
		self.numOfFollowersEdit.textChanged[str].connect(self.check_if_custom_values)
		self.numOfFriendsEdit.textChanged[str].connect(self.check_if_custom_values)
		self.numOfFavoritesEdit.textChanged[str].connect(self.check_if_custom_values)
		self.sentimentEdit.textChanged[str].connect(self.check_if_custom_values)
		self.datetimeEdit.textChanged[str].connect(self.check_if_custom_values)
		self.mentionsEdit.textChanged[str].connect(self.check_if_custom_values)
		self.hashtagsEdit.textChanged[str].connect(self.check_if_custom_values)
		self.entitiesCountEdit.textChanged[str].connect(self.check_if_custom_values)

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
		model_path = '../models/65InpLinReg-0408-2107'
		out_size = 1
		hidden_size = 32
		inp_size = 65 # "log_#Retweets" will be dropped when creating dataset hence 66 - 1 = 65
		self.model = load_model(model_path, inp_size, hidden_size, out_size)

	def randomize(self):
		self.value_selected_randomly_from_dataset = True
		index = random.randint(0, len(self.data.index))
		data_point = self.data.iloc[index]
		self.tweet_id = str(data_point['Tweet Id'])
		self.username = str(data_point['Username'])

		# now = str(datetime.now().strftime('%a %b %d %H:%M:%S +0000 %Y')) # EEE MMM dd HH:mm:ss Z yyyy
		# set texts into various line edits and labels
		self.numOfFollowersEdit.setText( str(data_point['#Followers']) )
		self.numOfFriendsEdit.setText( str(data_point['#Friends']) )
		self.numOfFavoritesEdit.setText( str(data_point['#Favorites']) )

		self.sentimentEdit.setText( str(data_point['Sentiment']) )
		self.datetimeEdit.setText( str(data_point['Timestamp']) )

		self.mentionsEdit.setText( str(data_point['Mentions']) )
		self.hashtagsEdit.setText( str(data_point['Hashtags']) )

		self.entitiesCountEdit.setText( str(len(data_point['Entities'].split(' '))) )
		self.value_selected_randomly_from_dataset = False

		self.trueValueIndex.setText(f'Data referenced. Index: {index}. Tweet Id: {self.tweet_id}.')
		self.true_value_label = str(data_point['#Retweets'])
		self._update_values(self.true_value_label, self.trueValueLabel)

	def check_if_custom_values(self):
		if self.value_selected_randomly_from_dataset: return

		# reset true label to 0 because no real data referenced. this is custom data
		self.trueValueIndex.setText(f'No real data referenced. You are entering custom data.')
		self.true_value_label = 0
		self._update_values("-", self.trueValueLabel)

	def predict(self):
		# read values
		self.values = self._read_values()

		# coerce datatypes
		self.input = coerce_datatype( dict(zip(self.headers, self.values)), self.mention_embeddings, self.hashtag_embeddings )

		# feed into model
		model_out = predict(self.model, self.input) # (true, pred)
		print(f"True value: {model_out[0]}. Prediction: {model_out[1]}.")

		# show result
		self.prediction_value_label = str( model_out[1] )
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
		# print(f"{label} set to {string}")

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
