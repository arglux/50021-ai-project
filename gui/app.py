import sys
import sympy
from pathlib import Path
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
		# self.latex_text = ""
		# self.file_path = ""

		# attach button to function
		# self.browseImageButton.clicked.connect(self.browse_image)
		# self.loadImageButton.clicked.connect(self.load_image)
		# self.translateButton.clicked.connect(self.translate_to_latex)
		# self.renderButton.clicked.connect(self.render_latex)

		# auto-complete feauture
		# self.filePathEdit.setText(self.file_path)

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
