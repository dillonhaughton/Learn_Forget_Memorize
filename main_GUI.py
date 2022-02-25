# Author: Dillon Haughton
# Date: March, 2020
# Written in Pyton3.9

# Version 1.0

'''
  Table of Contents
 -------------------
- Canvas and Tables and Custum Button
- Functions
- Updates
- File Navigation
- Alert Window
- Adding Material
- Testing
- Image
- Scoring
- Home Tab
- Change Settings
- Arrangment
- Adding
- Machine Learning
- Progress
- Main


'''
# Imports
import multiprocessing
#multiprocessing.set_start_method('forkserver', force=True)
#multiprocessing.set_start_method('forkserver', force=True)
#multiprocessing.freeze_support()
#multiprocessing.set_start_method('spawn')
# This works! No Seconday Pounts of entry
multiprocessing.freeze_support()
multiprocessing.set_start_method('spawn')

import sys
import os
import shutil
import pandas as pd
import numpy as np
import csv
import datetime
import shutil
import PyQt5.QtCore as Qt 
from PyQt5 import QtMultimedia
import PyQt5.QtWidgets as Wig
import PyQt5.QtGui as GUI
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import subprocess
import time
from tensorflow.keras.models import load_model
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import cm
from math import log10

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import playsound
import bs4
from bs4 import BeautifulSoup

# These were added may have problems here with spec file
from fuzzywuzzy import fuzz
from PIL import Image
import pytesseract

#from keras.models import load_model
#=========================================
#	 Canvas and Table and Custom Button
#=========================================

class Canvas(FigureCanvasQTAgg):
	def __init__(self, parent=None, width=5, height=4):
		fig = Figure(figsize=(width, height))
		self.axes = fig.add_subplot(111)
		super(Canvas, self).__init__(fig)

# Abstract Table output for the Display platform
class TableModel(Qt.QAbstractTableModel):
	# Trying to add color to rows that have more incorrect than correct
	def __init__(self, data):
		super(TableModel, self).__init__()
		self._data = data
		self.rowscolor = []
		try:
			
			for i in range(len(self._data)):
				correct = self._data.iloc[i, 2]
				incorrect = self._data.iloc[i, 3]
				if incorrect > correct:
					self.rowscolor += [i]
		except:
			pass			

	def data(self, index, role):
		#if not index.isValid():
		#	return None

		#if role == Qt.Qt.CheckStateRole:
		#	return None
		
		if role == Qt.Qt.DisplayRole:
			value = self._data.iloc[index.row(), index.column()]
			try:
				soup = BeautifulSoup(value, 'html.parser')
				value = str(soup.get_text()).lstrip()
			except:
				pass	

			return str(value)
		
		# Fill this with the indexs where Correct < Incorrect	
		# Slow becuase it has to calculate this everytime. Put it outside
		# and it should run faster
		
		
		
		if role == Qt.Qt.BackgroundRole and index.row() in self.rowscolor:
			return Qt.QVariant(GUI.QColor(Qt.Qt.red))


	def rowCount(self, index):
		return self._data.shape[0]

	def columnCount(self, index):
		return self._data.shape[1]
	
	def headerData(self, section, orientation, role):
		# section is the index of the column/row.
		if role == Qt.Qt.DisplayRole:
			if orientation == Qt.Qt.Horizontal:
				return str(self._data.columns[section])

			if orientation == Qt.Qt.Vertical:
				return str(self._data.index[section])

#------------------------------------------------------------------------
# Abstract Table output for the Display platform
# This may require some work!
class SearchModel(Qt.QAbstractTableModel):
	# Trying to add color to rows that have more incorrect than correct
	def __init__(self, data, search_for, classes, subject):
		super(SearchModel, self).__init__()
		self._data = data
		self.search_for = search_for
		self.rowscolor = []
		#try:
			
		for i in range(len(self._data)):
			term = self._data.iloc[i, 0]
			defi = self._data.iloc[i, 1]

			if term.split('.')[-1] == 'png':
				term = pytesseract.image_to_string(Image.open('{}/{}/{}/{}/{}'.format(path, classes, subject, 'images' , term)))

			if defi.split('.')[-1] == 'png':
				defi = pytesseract.image_to_string(Image.open('{}/{}/{}/{}/{}'.format(path, classes, subject, 'images', defi)))

			
			# Need more work here. Will keep images for now since cant maintain text information	
			ratio1 = fuzz.ratio(term.lower(), search_for.lower())
			ratio2 = fuzz.ratio(defi.lower(), search_for.lower())
			#print(ratio1)
			#print(ratio2)
			if ratio1 >= 30 or ratio2 >= 30: 
				self.rowscolor += [i]
			else:
				pass	
		#except:
		#	pass			

	def data(self, index, role):
		#if not index.isValid():
		#	return None

		#if role == Qt.Qt.CheckStateRole:
		#	return None
		
		if role == Qt.Qt.DisplayRole:
			value = self._data.iloc[index.row(), index.column()]
			try:
				soup = BeautifulSoup(value, 'html.parser')
				value = str(soup.get_text()).lstrip()
			except:
				pass	

			return str(value)
		
		# Fill this with the indexs where Correct < Incorrect	
		# Slow becuase it has to calculate this everytime. Put it outside
		# and it should run faster
		
		
		
		if role == Qt.Qt.BackgroundRole and index.row() in self.rowscolor:
			return Qt.QVariant(GUI.QColor(Qt.Qt.green))


	def rowCount(self, index):
		return self._data.shape[0]

	def columnCount(self, index):
		return self._data.shape[1]
	
	def headerData(self, section, orientation, role):
		# section is the index of the column/row.
		if role == Qt.Qt.DisplayRole:
			if orientation == Qt.Qt.Horizontal:
				return str(self._data.columns[section])

			if orientation == Qt.Qt.Vertical:
				return str(self._data.index[section])				


# Abstract Table output for the Display platform
class TableModel_norm(Qt.QAbstractTableModel):
	# Trying to add color to rows that have more incorrect than correct
	def __init__(self, data):
		super(TableModel_norm, self).__init__()
		self._data = data
		
					
	def data(self, index, role):
		
		if role == Qt.Qt.DisplayRole:
			value = self._data.iloc[index.row(), index.column()]
			
			return str(value)

	def rowCount(self, index):
		return self._data.shape[0]

	def columnCount(self, index):
		return self._data.shape[1]
	
	def headerData(self, section, orientation, role):
		# section is the index of the column/row.
		if role == Qt.Qt.DisplayRole:
			if orientation == Qt.Qt.Horizontal:
				return str(self._data.columns[section])

			if orientation == Qt.Qt.Vertical:
				return str(self._data.index[section])				

# Custom Button section
# Try to get this button to move like its being pressed when clicked
# Could in theory just replace it realy quick with an indented version.....
class MyButton(Wig.QPushButton):
	# OUTDATED
	def __init__(self, image, parent):
		super().__init__(parent)
		self.image = GUI.QPixmap(image)
		self.setMinimumSize(self.image.size())
		#self.setFixedSize(self.image.size())
		#self.setSizePolicy(Wig.QSizePolicy.Expanding, Wig.QSizePolicy.Expanding)
		self.setMask(self.image.createHeuristicMask())

	def paintEvent(self, event):
		qp = GUI.QPainter(self)
		qp.drawPixmap(Qt.QPoint(), self.image)


# New Button which will implement being pressed.....
# Need to fix sizing
# Need to make it so that it returns to normal after click
class PicButton(Wig.QAbstractButton):

	def __init__(self, pixmap, pixmap_hover, pixmap_pressed, parent=None):
		super().__init__(parent)
		self.pixmap = GUI.QPixmap(pixmap)
		self.pixmap_hover = GUI.QPixmap(pixmap_hover)
		self.pixmap_pressed = GUI.QPixmap(pixmap_pressed)

		#self.setSizePolicy(self.pixmap.size())

		self.setMask(self.pixmap.createHeuristicMask())
		self.setMask(self.pixmap_hover.createHeuristicMask())
		self.setMask(self.pixmap_pressed.createHeuristicMask())

		self.setCheckable(True)

	def paintEvent(self, event):
		pix = self.pixmap_hover if self.underMouse() else self.pixmap
		if self.isChecked():
			pix = self.pixmap_pressed
			
			self.setChecked(False)

		painter = GUI.QPainter(self)
		painter.drawPixmap(Qt.QPoint(), pix)


	def enterEvent(self, event):
		self.update()

	def leaveEvent(self, event):

		self.update()

	def sizeHint(self):
		return self.pixmap.size()		

#=========================================
#	           Functions
#=========================================
#---------------------------------------
def forgetting_curve(t,s):
	return np.exp(-t/s)
#---------------------------------------
def norm_forgetting_curve(F,base,s):
	field = np.arange(0,1000,0.1)
	A = F - base
	x =  (1.0*np.exp(-field/s)) + base
	return x
#---------------------------------------
def time_builder(curve):
	run = []
	count = 0
	for i in range(len(curve)):
		if curve[i] > curve[i-1]:
			run += [count]
		else:
			count += 0.1
			run += [count]
			
	return run      
#---------------------------------------
def s0(T):
	# shortest would be one can maybe go three days until first review
	# longest could be 10*10 or 20*10 depending on set size and contents
	# review should be very short then like half a day potentially
	# this could be very short

	# definitely too fast if a day has gone by you need to review the same day
	# that just wont work
	return 0.42*np.exp(-0.1*(T-1))

def s02(T):
	# This function will generate initial conditions based on length of the gen_frame
	# If one test was logged than 1 day, 2 tests logged than 2 and so on
	if T == 0:
		return 1
	else:
		return 2.8*T    

#---------------------------------------
def update_function1(forgetting_times):
	# Based on approximate parameters machine learning will replace in the future
	# Been replaced by Machine Learning
	return 10.857*forgetting_times + 2.3188

def update_function_read(forgetting_times):
	# This will update the material in the event the user does not want to use the 
	# machine learning or has not collected enough data
	path = os.getcwd()
	back_path = path.split('/')[:-1]
	back_path = '/'.join(back_path)

	try:
		des_values = pd.read_csv(back_path + '/time_lapse.csv', header=0, index_col=0)
		A = des_values['A'].values[0]
		B = des_values['B'].values[0]
	except:
		A = 10.857
		B = 2.3188
		time_frame = pd.DataFrame({'A':[A], 'B':[B]})
		time_frame.to_csv(back_path + '/time_lapse.csv', header=True, index=True)

	return A*forgetting_times + B

def update_function_adjust(forgetting_times, A, B):
	return A*forgetting_times + B

#---------------------------------------  
def exponential_test(t, alpha, b):
	return np.exp(alpha*t) + b

#---------------------------------------
def frame_to_listframe(dataframe):
	grob = []
	for i in range(len(dataframe)):
		row = dataframe.iloc[i,:].squeeze()
		grob += [row]

	return grob	


#=========================================
#	           Updaters
#=========================================			
# More speedy way of updating the curves of the files class structure
# since Goals need a update marker as well

# Utilizes the data_map
class update_version4:
	# OUTDATED No longer following curve data just using general
	def __init__(self):
		path = os.getcwd()
		back_path = path.split('/')[:-1]
		back_path = '/'.join(back_path)
		data_map = pd.read_csv(back_path + '/data_map.csv', header=0, index_col=0)
		data_map = data_map[data_map['L_F'] == 'F']
		moment = datetime.datetime.now()
		
		
		for i,j in data_map.iterrows():
			
			update_path = path + '/{}/{}'.format(j['Class'], j['Subject'])
			gen_update_path = update_path + '/general_data.csv'
			# Need to change here
			curve_update_path = update_path + '/curves_data.csv'

			read_gen = pd.read_csv(gen_update_path, header=0, index_col=0)
			read_curve = pd.read_csv(curve_update_path, header=None, index_col=None)

			date_change = pd.Series(datetime.datetime.strptime(read_gen.iloc[-1,1], '%Y-%m-%d %H:%M:%S.%f'))
			differentiator = moment - date_change
			days_gone = differentiator.dt.days
			days_gone_seconds = days_gone * 86400
			seconds_gone = differentiator.dt.seconds

			total_seconds = days_gone_seconds + seconds_gone
			incr_24 = total_seconds/(60*60*2.4)
			incr_24 = incr_24.astype(int).values[0]

			check_j = int(read_gen.iloc[-1,6])
			if read_gen.iloc[-1,5] == 'M':
				pass
			elif read_gen.iloc[-1,5] == 'L':
				pass	
			else:	
				if incr_24 > 0 and check_j != incr_24:
					F = 1
					base = float(read_gen.iloc[-1,3])
					s_value = float(read_gen.iloc[-1,4])
					update = norm_forgetting_curve(F, base, s_value)
					update = update[check_j+1: incr_24+1]
					curve = read_curve.values.flatten()

					series_up = pd.Series(update)

					if update[-1] == curve[-1]:
						pass
					else:
						curve = pd.concat([read_curve, series_up], axis=0)
						curve.to_csv(curve_update_path, header=False, index=False)
						read_gen.iloc[-1,6] = incr_24
						read_gen.to_csv(gen_update_path, index=True)

				else:
					pass



#=========================================
#	         File Navigation
#=========================================
def det_path(path):
	# Check to make sure in Pyinstaller
	if getattr(sys, 'frozen', False):
		root_path = sys._MEIPASS
	else:
		root_path = os.getcwd()

	path_to = os.path.join(root_path, path)
	return path_to

# Path is saved in settings but how do I access it? Maybe settings can be kept 
# in the main file everything else can be kept off site	

#------------------------------------------------
def access_settings(path, change = 0):
	# To read only or create set change to zero
	# To change set change to value of new goal
	if getattr(sys, 'frozen', False):
		root_path = sys._MEIPASS
	else:
		root_path = os.getcwd()
	root_path = root_path.split('/Data')[0]
	path_to = os.path.join(root_path, path)

	if change != 0:
		frame = pd.DataFrame({'Goals':[change[0]], 'tracker':[change[1]], 'date':[change[2]], 'maintain':[change[3]], 'text_size':[change[4]]})
		frame.to_csv(path_to)
	
	elif change == 0:
		try:
			frame = pd.read_csv(path_to, header=0, index_col=0)
		
		except:
			frame = pd.DataFrame({'Goals':[5], 'tracker':[0], 'date':[datetime.datetime.today().strftime('%Y-%m-%d')], 'maintain':[60], 'text_size':[13]})	
			frame.to_csv(path_to)

	return frame

#------------------------------------------------
# For Reading Folders
# Give 
class file_walker:
	def __init__(self, path):
		self.walker = list(os.walk(path))
		# list of classes called here
		self.classes = self.walker[0][1]

	def subjects(self, class_name):
		# Returns all subjects under class_name
		for i in self.walker:
			check = i[0].split('/')[-1]
			if check == class_name:
				# Error if same class and subject name
				if i[1] == ['images']:
					pass
				else:	
					subject = i[1]
			else:
				pass
		return subject
#------------------------------------------------
# For Reading Data Files 
class data_reader:
	def __init__(self, classes, subjects):
		self.classes = classes
		self.subjects = subjects
		# Subjects must be a list
		files = ['flashcards.csv',
				 'general_data.csv']

		path_to = os.getcwd() + '/{}/'.format(classes)

		self.flashcards_path = [path_to + '{}/{}'.format(i, files[0]) for i in subjects]
		self.general_data_path = [path_to + '{}/{}'.format(i, files[1]) for i in subjects]

	def determine_L_F(self):
		# This one will only return subjects seperated by Learning 
		# and forgetting all subjects in list
		Ls = []
		Fs = []
		for j,i in enumerate(self.general_data_path):
			call = pd.read_csv(i, index_col=0, header=0)
			LorF = call['L_F'].values[-1]
			if LorF == 'L':
				Ls += [self.subjects[j]]
			else:
				Fs += [self.subjects[j]]
		return Ls, Fs

	def determine_days2(self, s):
		# This function is more specific than the last one
		setting = access_settings(R'settings.csv', change=0)
		bench_post = float(setting['maintain'].values[0])/100
		#print(bench_post)
		for i in self.general_data_path:
			#print(i)
			call = pd.read_csv(i, index_col=0, header=0)
			#print(call)
			if call.iloc[-1,5] == 'F':
				now = datetime.datetime.today()
				then = call.iloc[-1,1]
				then = datetime.datetime.strptime(then, '%Y-%m-%d %H:%M:%S.%f')
				
				between = now - then
				between_real = between.days + between.seconds/86400
				time = np.arange(0, between_real, 0.1)
				# Want this as input for s value
				curve = forgetting_curve(time, s)
				final = curve[-1]
				# Want this as input for s value
				curve = norm_forgetting_curve(call.iloc[-1,2], call.iloc[-1,3], s)
				# Need to chop off first part as well. 
				curve = curve[curve > bench_post]
				curve = curve[curve < final]
				curve = len(curve)/10

				days = curve

			else:
				days = 0
		
		return days


	def todays(self):
		# OUTDATED going to remove Today portion
		frame = pd.DataFrame()
		for i,j in enumerate(self.general_data_path):
			call = pd.read_csv(j, index_col=0, header=0)
			if call.iloc[-1,5] == 'F':
				
				curve = pd.read_csv(self.curves_path[i], header=None).values
				current = curve[-1][0]
				f2 = pd.DataFrame({'C':[self.classes],'A':[call.iloc[-1,0]],'B':[current]})
				frame = frame.append(f2, ignore_index=True)
	
		return frame


	def plot_data(self):
		# Only works with one subject entry must be a list of one
		# will plot that subject

		gen = pd.read_csv(self.general_data_path[0], index_col=0, header=0)
		LorF = gen['L_F'].values[-1]

		if LorF == 'F':
			
			curve = pd.read_csv(self.curves_path[0],header=None).values
			t = time_builder(curve)
		else:
			curve = gen['F'].values
			t = [i for i in range(len(curve))]	
		return t, curve

	def plot_data2(self):
		# Generates the Plots without using the curve Plots portion if I can
		# remove the need than update function wont be needed either
		# Only works with one Data Point
		# Spit out differences in days and s values
		gen = pd.read_csv(self.general_data_path[0], index_col=0, header=0)
		LorF = gen['L_F'].values[-1]

		if LorF == 'F':

			gen = gen[gen['L_F'] == 'F']

			now = datetime.datetime.today()
			dates = list(gen['Date'].values)
			interp = []
			for i in dates:
				block = datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S.%f')
				interp += [block]

			dates = interp + [now]	
			diff_dates = np.diff(dates)
			days_interp = []
			for j in diff_dates:
				days_difference = j.days + j.seconds/86400
				days_interp += [days_difference]

			s_vals= gen['s_values'].values
			
			curve = np.array([0])
			for z,k in enumerate(days_interp):
				time = np.arange(0, k, 0.1)
				add_on = forgetting_curve(time, s_vals[z])
				
				curve = np.concatenate([curve ,add_on])
				
			t = time_builder(curve)	
			

		else:
			curve = gen['F'].values
			t = [i for i in range(len(curve))]

		return t, curve	
		

	def retrieve_flashcards(self):
		frame = pd.read_csv(self.flashcards_path[0], index_col=0, header=0)
		flashcards = frame['Term'].values
		definitions= frame['Definition'].values
		return flashcards, definitions


	def prepare_test(self):
		# Only works for one subject
		paths = [self.flashcards_path[0]]
		frames = [pd.read_csv(i, index_col=0, header=0) for i in paths]
		bottom_range = [len(i) for i in frames]
		new_top_range = []
		new_bottom_range = []
		for i,j in enumerate(bottom_range):
			for x in range(j):
				new_top_range += [[i, x]]

		np.random.shuffle(new_top_range)
		
		test = []
		for i in new_top_range:
			test += [frames[i[0]].iloc[i[1],:]]
		return test, new_top_range

	def number_of_questions(self):
		# Only works for one subject
		paths = [self.flashcards_path[0]]
		frames = [pd.read_csv(i, index_col=0, header=0) for i in paths]
		amounts = [len(i) for i in frames]

		amount = np.sum(amounts)
		return amount

	def progress(self):
		# Need to change here
		# Can Update Later
		gen_paths = self.general_data_path

		data = pd.DataFrame({'Subject':[], 'Test Score':[], 'Estimated Score':[]})
		for j,i in enumerate(gen_paths):
			try:
				gen_swoop = pd.read_csv(i, index_col=0, header=0)
				# Need to change here
				cur_swoop = pd.read_csv(self.curves_path[j], index_col=None, header=None)
				nam_bit = gen_swoop.iloc[-1,0]
				gen_bit = gen_swoop.iloc[-1,2]
				cur_bit = cur_swoop.values.flatten()[-1]
				piece = pd.DataFrame({'Subject':[nam_bit], 'Test Score':[float('{:.2f}'.format(gen_bit))], 'Estimated Score':float('{:.2f}'.format(cur_bit))})
				data = data.append(piece, ignore_index=True)
			except:
				pass
		return data	


	def progress2(self):
		gen_paths = self.general_data_path
		#print(gen_paths)
		data = pd.DataFrame({'Subject':[], 'Test Score':[], 'Estimated Score':[]})
		for j,i in enumerate(gen_paths):
			try:
				gen_swoop = pd.read_csv(i, index_col=0, header=0)

				nam_bit = gen_swoop.iloc[-1,0]
				gen_bit = gen_swoop.iloc[-1,2]
				date = gen_swoop.iloc[-1,1]
				date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')

				now = datetime.datetime.now()

				diff = now - date
				diff_days = diff.days + diff.seconds/86400
				time = np.arange(0, diff_days, 0.1)

				s = gen_swoop.iloc[-1,4]

				curve = forgetting_curve(time, s)
				estimated = curve[-1]

				piece = pd.DataFrame({'Subject':[nam_bit], 'Test Score':[float('{:.2f}'.format(gen_bit))], 'Estimated Score':float('{:.2f}'.format(estimated))})
				data = data.append(piece, ignore_index=True)
			except:
				pass
		return data			

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# For writing to data files
class data_writer:
	def __init__(self, classes, subject):
		# subject can be single entry for this one
		self.classes = classes
		self.subject = subject
		self.path = os.getcwd() + '/{}/{}'.format(classes, subject)
		self.files = list(os.walk(self.path))[0][2]

		files_name = ['flashcards.csv',
					  'general_data.csv']
		  
		self.flashcards_path = self.path + '/{}'.format(files_name[0])
		self.general_data_path = self.path + '/{}'.format(files_name[1])      
		
	def add_flashcards(self, term, definition, tag):
		# Encorporate the fact that this one now 
		# encompasses Flashcards, Diagrams, Drawings etc
		# Allow to recognize cards, images, and movies

		# Submit a Pull as well
		new_path = self.flashcards_path
		frame = pd.read_csv(new_path, index_col=0, header=0)
		new_frame = pd.DataFrame({'Term':[term],
								  'Definition':[definition],
								  'Correct':[0],
								  'Incorrect':[0]})
		frame = frame.append(new_frame, ignore_index=True)
		frame.to_csv(new_path, index=True)

		if tag[0] != 0:
			to_path = self.path + '/images/{}'.format(term)
			shutil.move(tag[0], to_path)
		else:
			pass

		if tag[1] != 0:
			to_path = self.path + '/images/{}'.format(definition)
			shutil.move(tag[1], to_path)
		else:
			pass


	def add_diagrams3(self, here):
		# Most recent version
		# Call Dialog to ask for image
		# New Formatting
		pull = Wig.QFileDialog.getOpenFileName(here, 'Question Select', '','(*png)')
		pull = str(pull[0])
		helper = subprocess.call(['open' , pull])
		time.sleep(2)
		pull2 = Wig.QFileDialog.getOpenFileName(here, 'Answer Select','','(*.png)')
		pull2 = str(pull2[0])

		name1 = pull.split('/')[-1]
		name2 = pull2.split('/')[-1]
		to_path = self.path + '/images/{}'.format(name1)
		to_correct_path = self.path + '/images/{}'.format(name2)
		shutil.move(pull, to_path)
		shutil.move(pull2, to_correct_path)

		new_path = self.flashcards_path
		frame = pd.read_csv(new_path, index_col=0, header=0)
		new_frame = pd.DataFrame({'image path':[name1],
								  'image correct path':[name2],
								  'correct':[0],
								  'incorrect':[0]})
		frame = frame.append(new_frame, ignore_index=True)
		frame.to_csv(new_path, index=True)
	
	def add_drawings3(self, here, image_label):
		# OUTDATED
		pull = Wig.QFileDialog.getOpenFileName(here, 'Question Select', '','(*png)')
		pull = str(pull[0])
		#helper = subprocess.call(['open' , pull])

		name = pull.split('/')[-1]

		to_path = self.path + '/images/{}'.format(name)
		shutil.move(pull, to_path)
		new_path = self.flashcards_path
		frame = pd.read_csv(new_path, index_col=0, header=0)
		new_frame = pd.DataFrame({'To Draw':[image_label],
								  'Correct Image':[name],
								  'correct':[0],
								  'incorrect':[0]})
		frame = frame.append(new_frame, ignore_index=True)
		frame.to_csv(new_path, index=True)

	def add_study_time(self, amount_time):
		# OUTDATED
		# Considering Outdated removing Test Window
		# First new row is created here
		new_path = self.general_data_path
		frame = pd.read_csv(new_path, index_col=0, header=0)
		new_frame = pd.DataFrame({'Subjects':[self.subject],
								  'Date':[datetime.datetime.now()],
								  'F':[0],
								  'base':[0],
								  's_values':[frame.iloc[-1,4]],
								  'L_F':[frame.iloc[-1,5]],
								  'j':[0],
								  'studied (min)':[amount_time]})

		frame = frame.append(new_frame, ignore_index=True)
		frame.to_csv(new_path, index=True)

	def add_L_F(self, bene):
		# This needs to be changed when machine learning comes along
		# This has screwed up alot if jumped from forgettingn to learning.

		# incorporate machine learning into this!!
		if bene == 'F':
			back_path = os.getcwd().split('/')[:-1]
			back_path = '/'.join(back_path)

			# Changing the Dataa Map Values
			data_map = pd.read_csv(back_path + '/data_map.csv', header=0, index_col=0)
			mask = (data_map['Class'] == self.classes) & (data_map['Subject'] == self.subject)
			data_map['L_F'][mask] = 'F'
			data_map.to_csv(back_path + '/data_map.csv', header=True, index=True)


			general_data_frame = pd.read_csv(self.general_data_path, header=0, index_col=0)
			general_data_frame.iloc[-1,5] = 'F'


			learning_len = len(general_data_frame[general_data_frame['L_F'] == 'L'])
			
			forgetting_len = len(general_data_frame[general_data_frame['L_F'] == 'F'])


			call = data_reader(self.classes, [self.subject])
			amount = call.number_of_questions()

			Fs = general_data_frame['F'].values[1:]
			
			if len(Fs) == 0:
				average_score = 0
				std_score = 0
			else:	
				average_score = np.mean(Fs)
				std_score = np.std(Fs)
			
			
			Ls = general_data_frame[general_data_frame['L_F'] == 'L']
			Fs = general_data_frame[general_data_frame['L_F'] == 'F']
			Ls_dates = Ls['Date'].values
			Fs_dates = Fs['Date'].values
			try:
				difference = datetime.datetime.strptime(Fs_dates[0], '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(Ls_dates[0], '%Y-%m-%d %H:%M:%S.%f')
				days_difference = difference.days + difference.seconds/86400
			except:
				days_difference = 0	
			

			dat_frame = general_data_frame['Date'].values
			dates_to = []
			for k in dat_frame:
				dates_to += [datetime.datetime.strptime(k, '%Y-%m-%d %H:%M:%S.%f')]
			
			if len(dates_to) > 1:
				differences = np.diff(dates_to)
				means_day = []
				for zz in differences:
					means_day += [zz.days + zz.seconds/86400]
				mean_diff = np.mean(means_day)
				std_diff = np.std(means_day)

			else:
				# If there is only one time entry which should be impossible but just in case
				mean_diff = 0
				std_diff = 0

			all_class = file_walker(os.getcwd()).classes
			all_class = pd.Series(all_class)
			all_class = all_class.sort_values()
			
			labels, uniques = all_class.factorize()
			
			crypt = pd.DataFrame({'labels':labels, 'uniques':uniques})
			crypt.to_csv(back_path + '/crypts.csv', index=True)
			decrypt = crypt[crypt['uniques']==self.classes]
			decrypt = decrypt['labels'].values[0]
			
			
				# Need to come back and load this one later, since this model was not built for
			try:	
				model = load_model(back_path + '/machine_model.h5')

				# Class names need to be turned into integers here
				
				info = np.array([decrypt, learning_len, forgetting_len, amount, average_score, std_score,
								 days_difference, mean_diff, std_diff])
				info = info.reshape(1,9)
				new_s_eq = model.predict(info)[0][0]
				
			except:
				
				# Need to update the update function
				try:
					new_s_eq = s02(len(general_data_frame))

				except:
					new_s_eq = 0.14	
	
			#read_curve = pd.read_csv(self.curves_path, header=None, index_col=None)
			#mach = pd.read_csv(self.mach_data_path, header=None, index_col=None)
			
			#new_row = pd.DataFrame({'Subjects':[self.subject],
			#						'Date':[str(datetime.datetime.now())],
			#						'F':[1],
			#						'base':[0],
			#						's_values':[new_s_eq],
			#						'L_F':['F'],
			#						'j':[0],
			#						'studied (min)':[0]})

			general_data_frame.iloc[-1,4] = new_s_eq

			#s1 = pd.Series([1])
			#mach = pd.Series([len(read_curve)])
			#read_curve = read_curve.append([s1], ignore_index=True)
			#general_data_frame = general_data_frame.append(new_row, ignore_index=True)
			# Need to change here
			#read_curve.to_csv(self.curves_path, index=False, header=False)
			general_data_frame.to_csv(self.general_data_path, index=True)
			#mach.to_csv(self.mach_data_path, index=False, header=False)

		elif bene == 'M':
			back_path = os.getcwd().split('/')[:-1]
			back_path = '/'.join(back_path)

			data_map = pd.read_csv(back_path + '/data_map.csv', header=0, index_col=0)
			mask = (data_map['Class'] == self.classes) & (data_map['Subject'] == self.subject)
			data_map['L_F'][mask] = 'M'
			data_map.to_csv(back_path + '/data_map.csv', header=True, index=True)

			# Need to change here
			#read_curve = pd.read_csv(self.curves_path, header=None, index_col=None)
			#mach = pd.read_csv(self.mach_data_path, header=None, index_col=None)
			frame = pd.read_csv(self.general_data_path, header=0, index_col=0)
			new_row = pd.DataFrame({'Subjects':[self.subject],
									'Date':[datetime.datetime.now()],
									'F':[1],
									'base':[0],
									's_values': [0.14],
									'L_F':['M'],
									'j':[0],
									'studied (min)':[0]})
			
			frame = frame.append(new_row, ignore_index=True)
			frame.to_csv(self.general_data_path, index=True)

		else:
			back_path = os.getcwd().split('/')[:-1]
			back_path = '/'.join(back_path)

			data_map = pd.read_csv(back_path + '/data_map.csv', header=0, index_col=0)
			mask = (data_map['Class'] == self.classes) & (data_map['Subject'] == self.subject)
			data_map['L_F'][mask] = 'L'
			data_map.to_csv(back_path + '/data_map.csv', header=True, index=True)

			# Need to change here
			#read_curve = pd.read_csv(self.curves_path, header=None, index_col=None)
			#mach = pd.read_csv(self.mach_data_path, header=None, index_col=None)
			frame = pd.read_csv(self.general_data_path, header=0, index_col=0)
			new_row = pd.DataFrame({'Subjects':[self.subject],
									'Date':[datetime.datetime.now()],
									'F':[0],
									'base':[0],
									's_values': [0.14],
									'L_F':['L'],
									'j':[0],
									'studied (min)':[0]})
			
			frame = frame.append(new_row, ignore_index=True)
			frame.to_csv(self.general_data_path, index=True)

			mach = open(self.mach_data_path, 'w')
			#data = pd.Series([0])
			# Need to change here
			#data.to_csv(self.curves_path, index=False)
				

	def completed_test(self, ctf, arrange, score):
		#-----------------------
		# Errors seen at zero with curve fitting
		
		if score == 0.0:
			score = 0.001
		else:
			pass
			

		back_path = os.getcwd().split('/')[:-1]
		back_path = '/'.join(back_path)

		flashcards = pd.DataFrame({'Term':[],'Definition':[],'Correct':[],'Incorrect':[]})

		for i in ctf:
			
			flashcards = flashcards.append(i, ignore_index=True)
			

		flashcards.to_csv(self.flashcards_path, index=True)
		
		general_data_frame = pd.read_csv(self.general_data_path, index_col=0, header=0)

		count_F = len(general_data_frame[general_data_frame['L_F'] == 'F'])

		# Need to change here
		#curves_data_frame = pd.read_csv(self.curves_path, index_col=None, header=None)

		if general_data_frame.iloc[-1,5] == 'L':
			# Not adding Curves Data for Learning Curves anymore

			#s1 = pd.Series([score])
			#curves_data_frame = curves_data_frame.append([s1], ignore_index=True)

			# Adding study time section here so that it doesnt mess with 
			# machine learning variables
			
			new_frame = pd.DataFrame({'Subjects':[self.subject],
									  'Date':[str(datetime.datetime.now())],
									  'F':[0],
									  'base':[0],
									  's_values':[general_data_frame.iloc[-1,4]],
									  'L_F':[general_data_frame.iloc[-1,5]],
									  'j':[0],
									  'studied (min)':[0]})

			general_data_frame = general_data_frame.append(new_frame, ignore_index=True)
			#curves_data_frame.to_csv(new_path, index=True)

			general_data_frame.iloc[-1,3] = general_data_frame.iloc[-1,2]
			general_data_frame.iloc[-1,2] = score
			general_data_frame.iloc[-1,6] = 0

			general_data_frame.to_csv(self.general_data_path, index=True)
			# Need to change here
			#curves_data_frame.to_csv(self.curves_path, index=False, header=False)
		
		else:
			#---------------------------
			# Want a BOOST Option here!
			#---------------------------
			
			# Need to drop and then put back last row of general dataframe since it
			# cant be used in this cycle for loading data

			# this will correctly update the curve
			# Need to change here

			# This needs to be altered here to account for no more curve

			date  = datetime.datetime.strptime(general_data_frame.iloc[-1,1], '%Y-%m-%d %H:%M:%S.%f')
			now   = datetime.datetime.today()

			diff_dates = now - date
			diff_dates = len(np.arange(0, (diff_dates.days + diff_dates.seconds/86400), 0.1)) * 0.1
			#print(diff_dates)
			x_data = [0, diff_dates]
			y_data = [1, score]
			
			
			#curves_data_values = curves_data_frame.values.flatten()
			#back_tracked = curves_data_values[::-1]
			#last_mark = len(back_tracked) - np.argmax(back_tracked) - 1
			#second_x = len(curves_data_values[last_mark:])*0.1
			#x_data = [0, second_x]
			#y_data = [1, score]

			# For days longer than 20 this may start to generate issues.

			if diff_dates > 20:
				p0 = 100
			
			else:
				p0 = 20	

			popt = curve_fit(forgetting_curve, x_data, y_data, p0=p0)
			#print(general_data_frame.iloc[-1,4])
			general_data_frame.iloc[-1,4] = popt[0][0]
			#print(general_data_frame.iloc[-1,4])
			
			# This portion I will keep just in case but it will be deleted later
			#try:
			#	mach = pd.read_csv(self.mach_data_path, index_col=None, header=None)
			#	mach = mach.append([popt[0][0]], ignore_index=True)
			#except:
			#	mach = pd.Series([popt[0][0]])	
			#mach.to_csv(self.mach_data_path, index=False, header=False)

			#x_range = np.arange(0, second_x, 0.1)
			#new_curve = forgetting_curve(x_range,popt[0][0])
			#s1 = pd.Series(new_curve)

			# Need to change here
			#fixer = curves_data_values[:last_mark].copy()
			#curves_data_values = np.append(fixer, new_curve)
			#curves_data_values[last_mark:] = new_curve
			#curves_data_values = np.append(curves_data_values,[1])
			#s2 = pd.Series(curves_data_values)
			#curves_data_frame  = s2.append([s1], ignore_index=True)

			
			# Run through this time in order to get values for previous guess in machine map
			#--------------------------------------------------------------------------------
			
			# This region build new mach_learning data file
			# FIX
			try:
				# If file does exist it will be picked up here
				maching = pd.read_csv(back_path + '/mach_data_map.csv', header=0, index_col=0)
			except:
				# If this file doesnt exist it will be built here

				maching = pd.DataFrame({'class':[], 'learning_length':[], 'forgetting_length':[], '#_cards':[], 
										'ave_score':[], 'std_score':[], 'time_to_learn':[], 'ave_time':[], 'std_time':[],
										's_values':[]})

			# Does not record information into machine learning if score is 1 or 0 since these tend to cause problems	
			if score == 1 or score == 0:
				pass
			else:		

			# Lots of calculatory redundancys
				learning_len = len(general_data_frame[general_data_frame['L_F'] == 'L'])
				forgetting_len = len(general_data_frame[general_data_frame['L_F'] == 'F'])

				call = data_reader(self.classes, [self.subject])
				amount = call.number_of_questions()

				Fs = general_data_frame['F'].values[1:]
				average_score = np.mean(Fs)
				std_score = np.std(Fs)

				Ls = general_data_frame[general_data_frame['L_F'] == 'L']
				Fs = general_data_frame[general_data_frame['L_F'] == 'F']
				Ls_dates = Ls['Date'].values
				Fs_dates = Fs['Date'].values
				difference = datetime.datetime.strptime(Fs_dates[0], '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(Ls_dates[0], '%Y-%m-%d %H:%M:%S.%f')
				days_difference = difference.days + difference.seconds/86400

				dat_frame = general_data_frame['Date'].values
				dates_to = []
				for k in dat_frame:
					dates_to += [datetime.datetime.strptime(k, '%Y-%m-%d %H:%M:%S.%f')]
			
				try:
					differences = np.diff(dates_to)
					means_day = []
					for zz in differences:
						means_day += [zz.days + zz.seconds/86400]
					mean_diff = np.mean(means_day)
					std_diff = np.std(means_day)

				except:
				# If there is only one time entry which should be impossible but just in case
					mean_diff = 0
					std_diff = 0
				
				mach_data_new = pd.DataFrame({'class':[self.classes], 'learning_length':[learning_len], 
											  'forgetting_length':[forgetting_len], '#_cards':[amount],
											  'ave_score':[average_score], 'std_score':[std_score],
											  'time_to_learn':[days_difference], 'ave_time':[mean_diff], 'std_time':[std_diff],
											  's_values':[popt[0][0]]})
				maching = maching.append(mach_data_new, ignore_index=True)
				maching.to_csv(back_path + '/mach_data_map.csv', index=True)

			# Need to run through it again in order to guess next value
			#-------------------------------------------------------------------
			new_frame = pd.DataFrame({'Subjects':[self.subject],
									  'Date':[str(datetime.datetime.now())],
									  'F':[0],
									  'base':[0],
									  's_values':[general_data_frame.iloc[-1,4]],  # Why is this here?
									  'L_F':[general_data_frame.iloc[-1,5]],
									  'j':[0],
									  'studied (min)':[0]})

			general_data_frame = general_data_frame.append(new_frame, ignore_index=True)

			general_data_frame.iloc[-1,3] = 0
			general_data_frame.iloc[-1,2] = score
			general_data_frame.iloc[-1,6] = 0

			learning_len = len(general_data_frame[general_data_frame['L_F'] == 'L'])
			forgetting_len = len(general_data_frame[general_data_frame['L_F'] == 'F'])

			call = data_reader(self.classes, [self.subject])
			amount = call.number_of_questions()

			Fs = general_data_frame['F'].values[1:]
			average_score = np.mean(Fs)
			std_score = np.std(Fs)

			Ls = general_data_frame[general_data_frame['L_F'] == 'L']
			Fs = general_data_frame[general_data_frame['L_F'] == 'F']
			Ls_dates = Ls['Date'].values
			Fs_dates = Fs['Date'].values
			difference = datetime.datetime.strptime(Fs_dates[0], '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(Ls_dates[0], '%Y-%m-%d %H:%M:%S.%f')
			days_difference = difference.days + difference.seconds/86400

			dat_frame = general_data_frame['Date'].values
			dates_to = []
			for k in dat_frame:
				dates_to += [datetime.datetime.strptime(k, '%Y-%m-%d %H:%M:%S.%f')]
			
			try:
				differences = np.diff(dates_to)
				means_day = []
				for zz in differences:
					means_day += [zz.days + zz.seconds/86400]
				mean_diff = np.mean(means_day)
				std_diff = np.std(means_day)

			except:
				# If there is only one time entry which should be impossible but just in case
				mean_diff = 0
				std_diff = 0

			all_class = file_walker(os.getcwd()).classes
			all_class = pd.Series(all_class)
			all_class = all_class.sort_values()
			
			labels, uniques = all_class.factorize()
			
			crypt = pd.DataFrame({'labels':labels, 'uniques':uniques})
			crypt.to_csv(back_path + '/crypts.csv', index=True)
			decrypt = crypt[crypt['uniques']==self.classes]
			decrypt = decrypt['labels'].values[0]
			
			
				# Need to come back and load this one later, since this model was not built for
			try:	
				model = load_model(back_path + '/machine_model.h5')

				# Class names need to be turned into integers here
				
				info = np.array([decrypt, learning_len, forgetting_len, amount, average_score, std_score,
								 days_difference, mean_diff, std_diff])
				info = info.reshape(1,9)
				new_s_eq = model.predict(info)[0][0]

			
				general_data_frame.iloc[-1,4] = new_s_eq

			except:
				
				# Need to update the update function
				general_data_frame.iloc[-1,4] = update_function_read(count_F)	

					
			general_data_frame.to_csv(self.general_data_path, index=True)
			# Need to change here
			#s2.to_csv(self.curves_path, index=False, header=False)

#-------------------------------------------------------------------------------------------------
class home_reader2:
	def __init__(self):
		self.path = os.getcwd()
		back_path = path.split('/')[:-1]
		back_path = '/'.join(back_path)
		self.data_map = pd.read_csv(back_path + '/data_map.csv', header=0, index_col=0)

	def everything(self):
		# Retrieves everything to be displayed by home screen. 
		learning = pd.DataFrame()
		for i,j in self.data_map.iterrows():
			work_with = 'general_data.csv'
			update_path = self.path + '/{}/{}/{}'.format(j['Class'], j['Subject'],work_with)

			frame = pd.read_csv(update_path, index_col=0, header=0)
			last_line = frame.iloc[-1:]
			F = last_line['F'].values[0]

				#value = ' -- '.join(disect)
			append_frame = pd.DataFrame({'A':[j['Class']], 'B':[j['Subject']], 'C':[F]})
			learning = learning.append(append_frame, ignore_index=True)

		return learning

	def learning(self):
		self.data_map = self.data_map[self.data_map['L_F'] == 'L']

		learning = pd.DataFrame()
		for i,j in self.data_map.iterrows():
			work_with = 'general_data.csv'
			update_path = self.path + '/{}/{}/{}'.format(j['Class'], j['Subject'],work_with)

			frame = pd.read_csv(update_path, index_col=0, header=0)
			last_line = frame.iloc[-1:]
			F = last_line['F'].values[0]

				#value = ' -- '.join(disect)
			append_frame = pd.DataFrame({'A':[j['Class']], 'B':[j['Subject']], 'C':[F]})
			learning = learning.append(append_frame, ignore_index=True)

		return learning

	def today(self):
		self.data_map = self.data_map[self.data_map['L_F'] == 'F']

		listed = pd.DataFrame()
		for i,j in self.data_map.iterrows():
			# Need to change here
			curve_with = 'general_data.csv'
			curve_path = self.path + '/{}/{}/{}'.format(j['Class'], j['Subject'],curve_with)


			frame = pd.read_csv(curve_path, index_col=0, header=0)
			s_values = frame.iloc[-1,4]
			dates    = datetime.datetime.strptime(frame.iloc[-1,1], '%Y-%m-%d %H:%M:%S.%f')
			now      = datetime.datetime.today()

			difference = now - dates
			days_difference = difference.days + difference.seconds/86400

			value = forgetting_curve(days_difference, s_values)


			#last_dot = frame.iloc[-1:].values[0][0]
			
			#value  = ' -- '.join(disect)
			append_frame = pd.DataFrame({'A':[j['Class']],'B':[j['Subject']] ,'C':[value]})
			listed = listed.append(append_frame, ignore_index=True)
			

		return listed	


	def forgetting_test_score(self):
		self.data_map = self.data_map[self.data_map['L_F'] == 'F']

		listed = pd.DataFrame()
		for i,j in self.data_map.iterrows():
			work_with = 'general_data.csv'
			update_path = self.path + '/{}/{}/{}'.format(j['Class'], j['Subject'],work_with)
			framer = pd.read_csv(update_path, index_col=0, header=0)

			framer = framer.iloc[-1:]
			F = framer['F'].values[0]
			
			#value  = ' -- '.join(disect)
			append_frame = pd.DataFrame({'A':[j['Class']], 'B':[j['Subject']], 'C':[F]})
			listed = listed.append(append_frame, ignore_index=True)
				
		return listed

	def memorized(self):
		self.data_map = self.data_map[self.data_map['L_F'] == 'M']

		listed = pd.DataFrame()
		for i,j in self.data_map.iterrows():
			work_with = 'general_data.csv'
			update_path = self.path + '/{}/{}/{}'.format(j['Class'], j['Subject'],work_with)
			framer = pd.read_csv(update_path, index_col=0, header=0)

			framer = framer.iloc[-1:]
			F = framer['F'].values[0]
			
			#value  = ' -- '.join(disect)
			append_frame = pd.DataFrame({'A':[j['Class']], 'B':[j['Subject']], 'C':[F]})
			listed = listed.append(append_frame, ignore_index=True)
				
		return listed

#=========================================
#			   ALERT Window
#=========================================	
class alert_win(Wig.QMainWindow):
	def __init__(self, text):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 500, 500
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle('Edit Table') # Window Title
		self.window = ALERT(text)
		self.setCentralWidget(self.window)

class ALERT(Wig.QWidget):
	def __init__(self, text):
		super().__init__()
		vlayout = Wig.QVBoxLayout(self)

		label = Wig.QLabel(text)
		label.setAlignment(Qt.Qt.AlignCenter)

		vlayout.addWidget(label)

#=========================================
#	         Adding Material
#=========================================

class adding_material(Wig.QMainWindow):
	def __init__(self, classes, subject):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 500, 1000
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Adding Material") # Window Title
		self.window = add1(classes, subject)
		self.setCentralWidget(self.window)


class add1(Wig.QWidget):
	def __init__(self, classes, subject):
		super().__init__()

		self.tag = [0,0]

		self.classes = classes
		self.subject = subject
		vlayout = Wig.QVBoxLayout(self)
		hlayout = Wig.QHBoxLayout()
		grid1 = Wig.QGridLayout()

		current_class_lab = Wig.QLabel('Class: {} | Subject: {}'.format(classes, subject),self)
		current_class_lab.setAlignment(Qt.Qt.AlignCenter)
		current_class_lab.setFixedHeight(50)
	

		self.term_lab = Wig.QLabel('Term')
		self.term_lab.setFixedHeight(20)
		self.term = Wig.QTextEdit(self)
		self.term.setFixedHeight(100)
		self.definition_lab = Wig.QLabel('Definition')
		self.definition_lab.setFixedHeight(20)
		self.definition = Wig.QTextEdit(self)

		file_button1 = Wig.QPushButton('...')
		file_button1.clicked.connect(self.add_file_button1)
		file_button2 = Wig.QPushButton('...')
		file_button2.clicked.connect(self.add_file_button2)

		add_button = Wig.QPushButton('Add Material')
		add_button.clicked.connect(self.send_material)

		shortcut = Wig.QShortcut(GUI.QKeySequence("Shift+Right"),self)
		shortcut.activated.connect(self.send_material)

		shortcut2 = Wig.QShortcut(GUI.QKeySequence("Ctrl+B"),self)
		shortcut2.activated.connect(self.bold_text)


		vlayout.addWidget(current_class_lab)

		grid1.addWidget(self.term_lab,0,0)
		grid1.addWidget(self.term,1,0)
		grid1.addWidget(file_button1,1,1)
		grid1.addWidget(self.definition_lab,2,0)
		grid1.addWidget(self.definition,3,0,2,1)
		grid1.addWidget(file_button2,3,1)
		
		vlayout.addLayout(grid1)
		vlayout.addWidget(add_button)

	def bold_text(self):
		self.definition.setFontWeight(GUI.QFont.Bold)
		#self.definition.setFontSize(20)
		self.term.setFontWeight(GUI.QFont.Bold)
		#self.term.setFontSize(20)




	def send_material(self):

		check1 = str(self.term.toPlainText())
		check2 = str(self.definition.toPlainText())

		if check1 != '' or check2 != '':

			call = data_writer(self.classes, self.subject)
			# Switching to html to get bold formatting
			if self.tag != [0,0]:
				call.add_flashcards(self.term.toPlainText(), self.definition.toPlainText(), self.tag)
			else:
				call.add_flashcards(self.term.toHtml(), self.definition.toHtml(), self.tag)
			self.term.clear()
			self.term.repaint()
			self.definition.clear()
			self.definition.repaint()

			self.tag = [0,0]
			self.term.setFontWeight(GUI.QFont.Normal)
			self.definition.setFontWeight(GUI.QFont.Normal)

		else:
			self.w = alert_win('Error: Not enough content')
			self.w.show()



	def add_file_button1(self):
		# need to check this out
		pull = Wig.QFileDialog.getOpenFileName(self, 'Question Select', '','(*png *.mov *.mp4 *.mp3)')
		pull = str(pull[0])

		name = pull.split('/')
		
		self.term.setText(name[-1])

		self.tag[0] = pull

	def add_file_button2(self):
		# need to check this out
		pull = Wig.QFileDialog.getOpenFileName(self, 'Question Select', '','(*png *.mov *.mp4 *mp3)')
		pull = str(pull[0])

		name = pull.split('/')
		
		self.definition.setText(name[-1])

		self.tag[1] = pull	
		

# Updates happen here
path = det_path(R'Data')
try:
	os.chdir(path)
except:
	os.mkdir(path)
	os.chdir(path)


#=========================================
#	              Testing
#=========================================	
path = os.getcwd()

class test_screen_pop(Wig.QMainWindow):
	def __init__(self, classes, subjects):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 1200, 1000
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Testing") # Window Title
		self.window = test1(classes, subjects)
		self.setCentralWidget(self.window)

class test1(Wig.QWidget):
	def __init__(self, classes, subjects):
		super().__init__()
		grid1 = Wig.QGridLayout()
		grid2 = Wig.QGridLayout()
		grid3 = Wig.QGridLayout()
		vlay  = Wig.QVBoxLayout(self)

		self.call_classes = classes
		self.call_subject = subjects

		call = data_reader(classes, [subjects])
		#Update
		call2 = call.prepare_test()
		self.test = call2[0]
		self.rearrange = call2[1]
		self.total_amount = len(self.test)
		self.on = 0


		self.class_lab = Wig.QLabel('Test For {}-{}: {}/{}'.format(classes, subjects, self.on, self.total_amount))
		self.class_lab.setFixedHeight(10)
		self.class_lab.setAlignment(Qt.Qt.AlignCenter)

		self.start_butt = Wig.QPushButton('Start')
		self.start_butt.clicked.connect(self.start_test)

		self.screen = Wig.QLabel()
		self.settings = access_settings(R'settings.csv', change=0)
		font = int(self.settings['text_size'].values[0])

		self.screen.setStyleSheet(' font-size: {}px; '.format(font))

		self.screen.setWordWrap(True)
		self.screen.setAlignment(Qt.Qt.AlignCenter)
		self.screen.setSizePolicy(Wig.QSizePolicy.Expanding, Wig.QSizePolicy.Expanding)

		scroll = Wig.QScrollArea()
		scroll.setWidget(self.screen)
		scroll.setWidgetResizable(True)

		self.flip_button       = Wig.QPushButton('FLIP')
		self.flip_button.clicked.connect(self.flipper)
		self.flip_button.setFixedHeight(20)

		self.num_correct = Wig.QLineEdit(self)
		self.num_correct.setFixedHeight(20)
		self.num_correct.setFixedWidth(100)
		self.num_incorrect = Wig.QLineEdit(self)
		self.num_incorrect.setFixedHeight(20)
		self.num_incorrect.setFixedWidth(100)

		incorrect  = Wig.QPushButton('Wrong')
		incorrect.setFixedWidth(100)
		incorrect.clicked.connect(self.incorrect_store)

		shortcut1 = Wig.QShortcut(GUI.QKeySequence("Shift+Left"),self)
		shortcut1.activated.connect(self.incorrect_store)

		correct    = Wig.QPushButton('Correct')
		correct.setFixedWidth(100)
		correct.clicked.connect(self.correct_store)

		shortcut2 = Wig.QShortcut(GUI.QKeySequence("Shift+Right"),self)
		shortcut2.activated.connect(self.correct_store)

		shortcut3 = Wig.QShortcut(GUI.QKeySequence("Shift+Down"),self)
		shortcut3.activated.connect(self.flipper)

		grid1.addWidget(self.class_lab,0,0)
		grid1.addWidget(self.start_butt,1,0)

		grid3.addWidget(incorrect,0,0)
		grid3.addWidget(scroll,0,1)
		grid3.addWidget(correct,0,2)

		grid2.addWidget(self.num_incorrect,0,0)
		grid2.addWidget(self.flip_button,0,1)
		grid2.addWidget(self.num_correct,0,2)

		vlay.addLayout(grid1)
		vlay.addLayout(grid3)
		vlay.addLayout(grid2)

	def start_test(self):
		#try:
		if self.start_butt.text() == 'Start':
			self.frame = self.test
			#print(type(self.frame[0]))
			self.score  = 0
			self.anti_score = 0
			self.marker = 0
			self.flip   = 0

			bottom_side = list(self.frame[self.marker].index)[1]
			self.flip_button.setText(bottom_side)
			self.flip_button.repaint()
				# Now I need the screen to recognize pdfs based on path
			if self.frame[self.marker].iloc[0].split('.')[-1] == 'png':
				space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[0])

				pixmap = GUI.QPixmap(space)
				pixmap = pixmap.scaled(800, 800, aspectRatioMode=1)
				self.screen.setPixmap(pixmap)
				self.screen.setScaledContents(True)
				self.screen.repaint()

			elif self.frame[self.marker].iloc[0].split('.')[-1] == 'mov' or self.frame[self.marker].iloc[0].split('.')[-1] == 'mp4':
				space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[0])
				helper = subprocess.call(['open' , space])

			elif self.frame[self.marker].iloc[0].split('.')[-1] == 'mp3' or self.frame[self.marker].iloc[0].split('.')[-1] == 'MP3':
				space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[0])
				playsound.playsound(space)
				
			else:	
				self.screen.setText(self.frame[self.marker].iloc[0])
				self.screen.repaint()
			self.on += 1	
			self.class_lab.setText('Test For {}-{}: {}/{}'.format(self.call_classes, self.call_subject, self.on, self.total_amount))	
			self.start_butt.setText('Score')
			self.start_butt.repaint()
		else:
				
				# Here is the scoring section
			
			writer_call = data_writer(self.call_classes, self.call_subject)
			writer_call.completed_test(self.test, self.rearrange, self.score/self.anti_score)
			self.close()
			self.screen.repaint()
			

	def flipper(self):
		if self.flip == 0:
			if self.frame[self.marker].iloc[1].split('.')[-1] == 'png':
					space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[1])

					pixmap = GUI.QPixmap(space)
					pixmap = pixmap.scaled(800, 800, aspectRatioMode=1)
					self.screen.setPixmap(pixmap)
					self.screen.setScaledContents(True)
					self.screen.repaint()

			elif self.frame[self.marker].iloc[1].split('.')[-1] == 'mov' or self.frame[self.marker].iloc[1].split('.')[-1] == 'mp4':
				space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[1])
				helper = subprocess.call(['open' , space])	

			elif self.frame[self.marker].iloc[0].split('.')[-1] == 'mp3' or self.frame[self.marker].iloc[1].split('.')[-1] == 'MP3':
				# Problems with different langauge paths
				space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[1])
				playsound.playsound(space)	

			else:	
				self.screen.setText(self.frame[self.marker].iloc[1])
				self.screen.repaint()

			top_side = list(self.frame[self.marker].index)[0]
			self.flip_button.setText(top_side)
			self.flip_button.repaint()
			self.flip = 1

		else:
			if self.frame[self.marker].iloc[0].split('.')[-1] == 'png':
					space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[0])

					pixmap = GUI.QPixmap(space)
					pixmap = pixmap.scaled(800, 800, aspectRatioMode=1)
					self.screen.setPixmap(pixmap)
					self.screen.setScaledContents(True)
					self.screen.repaint()

			elif self.frame[self.marker].iloc[0].split('.')[-1] == 'mov' or self.frame[self.marker].iloc[0].split('.')[-1] == 'mp4':
				space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[0])
				helper = subprocess.call(['open' , space])

			elif self.frame[self.marker].iloc[0].split('.')[-1] == 'mp3' or self.frame[self.marker].iloc[0].split('.')[-1] == 'MP3':
				space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[0])
				playsound.playsound(space)	
						
			else:	
				self.screen.setText(self.frame[self.marker].iloc[0])
				self.screen.repaint()
			bottom_side = list(self.frame[self.marker].index)[1]
			self.flip_button.setText(bottom_side)
			self.flip_button.repaint()
			self.flip = 0
				
	def incorrect_store(self):
		if self.screen.text() == 'Finished':
			pass
		else:	
			self.flip = 0
			# This section is to set up the points
			try:
				if self.flip_button.text() == 'question' or self.flip_button.text() == 'answer':
					# OUTDATED no questions or answers anymore
					n_i = int(self.num_incorrect.text())
					n_c = int(self.num_correct.text())
					self.frame[self.marker].iloc[3] = self.frame[self.marker].iloc[3] + n_i
					self.frame[self.marker].iloc[2] = self.frame[self.marker].iloc[2] + n_c
				else:	
					n_i = int(self.num_incorrect.text())
					n_c = int(self.num_correct.text())
					self.frame[self.marker].iloc[3] = self.frame[self.marker].iloc[3] + n_i
					self.frame[self.marker].iloc[2] = self.frame[self.marker].iloc[2] + n_c
					self.score = self.score + n_c
					self.anti_score = self.anti_score + n_i

					self.num_incorrect.clear()
					self.num_correct.clear()
	

			except:
				if self.flip_button.text() == 'question' or self.flip_button.text() == 'answer':
					self.frame[self.marker].iloc[3] = self.frame[self.marker].iloc[3] + 1
				else:	
					self.frame[self.marker].iloc[3] = self.frame[self.marker].iloc[3] + 1
					self.anti_score = self.anti_score + 1
				

			# This section sets up the next card
			if self.marker >= len(self.frame)-1:
				self.screen.clear()
				self.screen.setText('Finished')
				self.class_lab.setText('Test For {}-{}: {}'.format(self.call_classes, self.call_subject, 'Done'))
				self.screen.repaint()
				#self.start_butt.click()

			else:	
				self.marker += 1
				self.screen.clear()
				if self.frame[self.marker].iloc[0].split('.')[-1] == 'png':
					space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[0])

					pixmap = GUI.QPixmap(space)
					pixmap = pixmap.scaled(800, 800, aspectRatioMode=1)
					self.screen.setPixmap(pixmap)
					self.screen.setScaledContents(True)
					self.screen.repaint()

				elif self.frame[self.marker].iloc[0].split('.')[-1] == 'mov' or self.frame[self.marker].iloc[0].split('.')[-1] == 'mp4':
					space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[0])
					helper = subprocess.call(['open' , space])

				elif self.frame[self.marker].iloc[0].split('.')[-1] == 'mp3' or self.frame[self.marker].iloc[0].split('.')[-1] == 'MP3':
					space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[0])
					playsound.playsound(space)

				else:	
					self.screen.setText(self.frame[self.marker].iloc[0])
					self.screen.repaint()

				self.on += 1
				self.class_lab.setText('Test For {}-{}: {}/{}'.format(self.call_classes, self.call_subject, self.on, self.total_amount))
			bottom_side = list(self.frame[self.marker].index)[1]
			self.flip_button.setText(bottom_side)
			self.flip_button.repaint()

	def correct_store(self):
		if self.screen.text() == 'Finished':
			pass
		else:	
			self.flip = 0
			try:
				if self.flip_button.text() == 'question' or self.flip_button.text() == 'answer':
					n_i = int(self.num_incorrect.text())
					n_c = int(self.num_correct.text())
					self.frame[self.marker].iloc[3] = self.frame[self.marker].iloc[3] + n_i
					self.frame[self.marker].iloc[2] = self.frame[self.marker].iloc[2] + n_c
				else:	
					n_i = int(self.num_incorrect.text())
					n_c = int(self.num_correct.text())
					self.frame[self.marker].iloc[3] = self.frame[self.marker].iloc[3] + n_i
					self.frame[self.marker].iloc[2] = self.frame[self.marker].iloc[2] + n_c
					self.score = self.score + n_c
					self.anti_score = self.anti_score + n_i

					self.num_incorrect.clear()
					self.num_correct.clear()
				

			except:
				if self.flip_button.text() == 'question' or self.flip_button.text() == 'answer':
					self.frame[self.marker].iloc[2] = self.frame[self.marker].iloc[2] + 1	
				else:	
					self.frame[self.marker].iloc[2] = self.frame[self.marker].iloc[2] + 1
					self.score = self.score + 1
					self.anti_score = self.anti_score + 1
				

			if self.marker >= len(self.frame)-1:
				self.screen.clear()
				self.screen.setText('Finished')
				self.class_lab.setText('Test For {}-{}: {}'.format(self.call_classes, self.call_subject, 'Done'))
				self.screen.repaint()

			else:	
				self.marker += 1
				self.screen.clear()
				if self.frame[self.marker].iloc[0].split('.')[-1] == 'png':
					space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[0])

					pixmap = GUI.QPixmap(space)
					pixmap = pixmap.scaled(800, 800, aspectRatioMode=1)
					self.screen.setPixmap(pixmap)
					self.screen.setScaledContents(True)
					self.screen.repaint()

				elif self.frame[self.marker].iloc[0].split('.')[-1] == 'mov' or self.frame[self.marker].iloc[0].split('.')[-1] == 'mp4':
					space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[0])
					helper = subprocess.call(['open' , space])

				elif self.frame[self.marker].iloc[0].split('.')[-1] == 'mp3' or self.frame[self.marker].iloc[0].split('.')[-1] == 'MP3':
					space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[0])
					playsound.playsound(space)	

				else:	
					self.screen.setText(self.frame[self.marker].iloc[0])
					self.screen.repaint()

				self.on += 1
				self.class_lab.setText('Test For {}-{}: {}/{}'.format(self.call_classes, self.call_subject, self.on, self.total_amount))
			bottom_side = list(self.frame[self.marker].index)[1]
			self.flip_button.setText(bottom_side)
			self.flip_button.repaint()	


#-----------------------------------------
class test_for_giant_test(Wig.QMainWindow):
	def __init__(self, dataframe):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 1200, 1000
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Testing") # Window Title
		self.window = test_for_giant_test1(dataframe)
		self.setCentralWidget(self.window)

class test_for_giant_test1(Wig.QWidget):
	def __init__(self, dataframe):
		super().__init__()
		grid1 = Wig.QGridLayout()
		grid2 = Wig.QGridLayout()
		grid3 = Wig.QGridLayout()
		vlay  = Wig.QVBoxLayout(self)

		#self.call_classes = classes
		#self.call_subject = subjects
		#self.report = report

		#call = data_reader(classes, [subjects])
		#Update
		#call2 = call.prepare_test()
		self.test = dataframe
		
		#print(self.test)
		#self.rearrange = call2[1]
		self.total_amount = len(self.test)
		self.on = 0


		self.class_lab = Wig.QLabel('GIANT TEST: {}/{}'.format(self.on, self.total_amount))
		self.class_lab.setFixedHeight(10)
		self.class_lab.setAlignment(Qt.Qt.AlignCenter)

		self.start_butt = Wig.QPushButton('Start')
		self.start_butt.clicked.connect(self.start_test)

		self.screen = Wig.QLabel()
		self.settings = access_settings(R'settings.csv', change=0)
		font = int(self.settings['text_size'].values[0])

		self.screen.setStyleSheet(' font-size: {}px; '.format(font))

		self.screen.setWordWrap(True)
		self.screen.setAlignment(Qt.Qt.AlignCenter)
		self.screen.setSizePolicy(Wig.QSizePolicy.Expanding, Wig.QSizePolicy.Expanding)

		scroll = Wig.QScrollArea()
		scroll.setWidget(self.screen)
		scroll.setWidgetResizable(True)

		self.flip_button       = Wig.QPushButton('FLIP')
		self.flip_button.clicked.connect(self.flipper)
		self.flip_button.setFixedHeight(20)

		self.num_correct = Wig.QLineEdit(self)
		self.num_correct.setFixedHeight(20)
		self.num_correct.setFixedWidth(100)
		self.num_incorrect = Wig.QLineEdit(self)
		self.num_incorrect.setFixedHeight(20)
		self.num_incorrect.setFixedWidth(100)

		incorrect  = Wig.QPushButton('Wrong')
		incorrect.setFixedWidth(100)
		incorrect.clicked.connect(self.incorrect_store)

		shortcut1 = Wig.QShortcut(GUI.QKeySequence("Shift+Left"),self)
		shortcut1.activated.connect(self.incorrect_store)

		correct    = Wig.QPushButton('Correct')
		correct.setFixedWidth(100)
		correct.clicked.connect(self.correct_store)

		shortcut2 = Wig.QShortcut(GUI.QKeySequence("Shift+Right"),self)
		shortcut2.activated.connect(self.correct_store)

		shortcut3 = Wig.QShortcut(GUI.QKeySequence("Shift+Down"),self)
		shortcut3.activated.connect(self.flipper)

		grid1.addWidget(self.class_lab,0,0)
		grid1.addWidget(self.start_butt,1,0)

		grid3.addWidget(incorrect,0,0)
		grid3.addWidget(scroll,0,1)
		grid3.addWidget(correct,0,2)

		grid2.addWidget(self.num_incorrect,0,0)
		grid2.addWidget(self.flip_button,0,1)
		grid2.addWidget(self.num_correct,0,2)

		vlay.addLayout(grid1)
		vlay.addLayout(grid3)
		vlay.addLayout(grid2)

	def start_test(self):
		#try:
		if self.start_butt.text() == 'Start':
			self.frame = self.test
			self.score  = 0
			self.anti_score = 0
			self.marker = 0
			self.flip   = 0

			bottom_side = list(self.frame[self.marker].index)[3]
			self.flip_button.setText(bottom_side)
			self.flip_button.repaint()
				# Now I need the screen to recognize pdfs based on path
			if self.frame[self.marker].iloc[2].split('.')[-1] == 'png':
				space = '{}/{}/{}/images/{}'.format(path, self.frame[self.marker].iloc[0], self.frame[self.marker].iloc[1], self.frame[self.marker].iloc[2])

				pixmap = GUI.QPixmap(space)
				pixmap = pixmap.scaled(800, 800, aspectRatioMode=1)
				self.screen.setPixmap(pixmap)
				self.screen.setScaledContents(True)
				self.screen.repaint()

			elif self.frame[self.marker].iloc[2].split('.')[-1] == 'mov' or self.frame[self.marker].iloc[2].split('.')[-1] == 'mp4':
				space = '{}/{}/{}/images/{}'.format(path, self.frame[self.marker].iloc[0], self.frame[self.marker].iloc[1], self.frame[self.marker].iloc[2])
				helper = subprocess.call(['open' , space])

			elif self.frame[self.marker].iloc[2].split('.')[-1] == 'mp3' or self.frame[self.marker].iloc[2].split('.')[-1] == 'MP3':
				space = '{}/{}/{}/images/{}'.format(path, self.frame[self.marker].iloc[0], self.frame[self.marker].iloc[1], self.frame[self.marker].iloc[2])
				playsound.playsound(space)
				
			else:	
				self.screen.setText(self.frame[self.marker].iloc[2])
				self.screen.repaint()
			self.on += 1	

			self.class_lab.setText('GIANT TEST: {}/{}'.format(self.on, self.total_amount))	
			self.start_butt.setText('Score')
			self.start_butt.repaint()
		else:
				
				# Here is the scoring section
			
			final = self.score/self.anti_score
			self.class_lab.setText(str(final))	
			#self.close()
			#self.screen.repaint()
			

	def flipper(self):
		if self.flip == 0:
			if self.frame[self.marker].iloc[3].split('.')[-1] == 'png':
					space = '{}/{}/{}/images/{}'.format(path, self.frame[self.marker].iloc[0], self.frame[self.marker].iloc[1], self.frame[self.marker].iloc[3])

					pixmap = GUI.QPixmap(space)
					pixmap = pixmap.scaled(800, 800, aspectRatioMode=1)
					self.screen.setPixmap(pixmap)
					self.screen.setScaledContents(True)
					self.screen.repaint()

			elif self.frame[self.marker].iloc[3].split('.')[-1] == 'mov' or self.frame[self.marker].iloc[3].split('.')[-1] == 'mp4':
				space = '{}/{}/{}/images/{}'.format(path, self.call_classes, self.call_subject, self.frame[self.marker].iloc[1])
				helper = subprocess.call(['open' , space])	

			elif self.frame[self.marker].iloc[3].split('.')[-1] == 'mp3' or self.frame[self.marker].iloc[3].split('.')[-1] == 'MP3':
				# Problems with different langauge paths
				space = '{}/{}/{}/images/{}'.format(path, self.frame[self.marker].iloc[0], self.frame[self.marker].iloc[1], self.frame[self.marker].iloc[3])
				playsound.playsound(space)	

			else:	
				self.screen.setText(self.frame[self.marker].iloc[3])
				self.screen.repaint()

			top_side = list(self.frame[self.marker].index)[2]
			self.flip_button.setText(top_side)
			self.flip_button.repaint()
			self.flip = 1

		else:
			if self.frame[self.marker].iloc[2].split('.')[-1] == 'png':
					space = '{}/{}/{}/images/{}'.format(path, self.frame[self.marker].iloc[0], self.frame[self.marker].iloc[1], self.frame[self.marker].iloc[2])

					pixmap = GUI.QPixmap(space)
					pixmap = pixmap.scaled(800, 800, aspectRatioMode=1)
					self.screen.setPixmap(pixmap)
					self.screen.setScaledContents(True)
					self.screen.repaint()

			elif self.frame[self.marker].iloc[2].split('.')[-1] == 'mov' or self.frame[self.marker].iloc[2].split('.')[-1] == 'mp4':
				space = '{}/{}/{}/images/{}'.format(path, self.frame[self.marker].iloc[0], self.frame[self.marker].iloc[1], self.frame[self.marker].iloc[2])
				helper = subprocess.call(['open' , space])

			elif self.frame[self.marker].iloc[2].split('.')[-1] == 'mp3' or self.frame[self.marker].iloc[2].split('.')[-1] == 'MP3':
				space = '{}/{}/{}/images/{}'.format(path, self.frame[self.marker].iloc[0], self.frame[self.marker].iloc[1], self.frame[self.marker].iloc[2])
				playsound.playsound(space)	
						
			else:	
				self.screen.setText(self.frame[self.marker].iloc[2])
				self.screen.repaint()
			bottom_side = list(self.frame[self.marker].index)[3]
			self.flip_button.setText(bottom_side)
			self.flip_button.repaint()
			self.flip = 0
				
	def incorrect_store(self):
		if self.screen.text() == 'Finished':
			pass
		else:	
			self.flip = 0
			# This section is to set up the points
			try:
				if self.flip_button.text() == 'question' or self.flip_button.text() == 'answer':
					# OUTDATED no questions or answers anymore
					n_i = int(self.num_incorrect.text())
					n_c = int(self.num_correct.text())
					self.frame[self.marker].iloc[5] = self.frame[self.marker].iloc[5] + n_i
					self.frame[self.marker].iloc[4] = self.frame[self.marker].iloc[4] + n_c
				else:	
					n_i = int(self.num_incorrect.text())
					n_c = int(self.num_correct.text())
					self.frame[self.marker].iloc[5] = self.frame[self.marker].iloc[5] + n_i
					self.frame[self.marker].iloc[4] = self.frame[self.marker].iloc[4] + n_c
					self.score = self.score + n_c
					self.anti_score = self.anti_score + n_i

					self.num_incorrect.clear()
					self.num_correct.clear()
	

			except:
				if self.flip_button.text() == 'question' or self.flip_button.text() == 'answer':
					self.frame[self.marker].iloc[5] = self.frame[self.marker].iloc[5] + 1
				else:	
					self.frame[self.marker].iloc[5] = self.frame[self.marker].iloc[5] + 1
					self.anti_score = self.anti_score + 1
				

			# This section sets up the next card
			if self.marker >= len(self.frame)-1:
				self.screen.clear()
				self.screen.setText('Finished')
				self.class_lab.setText('GIANT TEST: {}/{}'.format(self.on, self.total_amount))	
				self.screen.repaint()
				#self.start_butt.click()

			else:	
				self.marker += 1
				self.screen.clear()
				if self.frame[self.marker].iloc[2].split('.')[-1] == 'png':
					space = '{}/{}/{}/images/{}'.format(path, self.frame[self.marker].iloc[0], self.frame[self.marker].iloc[1], self.frame[self.marker].iloc[2])

					pixmap = GUI.QPixmap(space)
					pixmap = pixmap.scaled(800, 800, aspectRatioMode=1)
					self.screen.setPixmap(pixmap)
					self.screen.setScaledContents(True)
					self.screen.repaint()

				elif self.frame[self.marker].iloc[2].split('.')[-1] == 'mov' or self.frame[self.marker].iloc[2].split('.')[-1] == 'mp4':
					space = '{}/{}/{}/images/{}'.format(path, self.frame[self.marker].iloc[0], self.frame[self.marker].iloc[1], self.frame[self.marker].iloc[2])
					helper = subprocess.call(['open' , space])

				elif self.frame[self.marker].iloc[2].split('.')[-1] == 'mp3' or self.frame[self.marker].iloc[2].split('.')[-1] == 'MP3':
					space = '{}/{}/{}/images/{}'.format(path, self.frame[self.marker].iloc[0], self.frame[self.marker].iloc[1], self.frame[self.marker].iloc[2])
					playsound.playsound(space)

				else:	
					self.screen.setText(self.frame[self.marker].iloc[2])
					self.screen.repaint()

				self.on += 1
				self.class_lab.setText('GIANT TEST: {}/{}'.format(self.on, self.total_amount))	
			bottom_side = list(self.frame[self.marker].index)[3]
			self.flip_button.setText(bottom_side)
			self.flip_button.repaint()

	def correct_store(self):
		if self.screen.text() == 'Finished':
			pass
		else:	
			self.flip = 0
			try:
				if self.flip_button.text() == 'question' or self.flip_button.text() == 'answer':
					n_i = int(self.num_incorrect.text())
					n_c = int(self.num_correct.text())
					self.frame[self.marker].iloc[5] = self.frame[self.marker].iloc[5] + n_i
					self.frame[self.marker].iloc[4] = self.frame[self.marker].iloc[4] + n_c
				else:	
					n_i = int(self.num_incorrect.text())
					n_c = int(self.num_correct.text())
					self.frame[self.marker].iloc[5] = self.frame[self.marker].iloc[5] + n_i
					self.frame[self.marker].iloc[4] = self.frame[self.marker].iloc[4] + n_c
					self.score = self.score + n_c
					self.anti_score = self.anti_score + n_i

					self.num_incorrect.clear()
					self.num_correct.clear()
				

			except:
				if self.flip_button.text() == 'question' or self.flip_button.text() == 'answer':
					self.frame[self.marker].iloc[4] = self.frame[self.marker].iloc[4] + 1	
				else:	
					self.frame[self.marker].iloc[4] = self.frame[self.marker].iloc[4] + 1
					self.score = self.score + 1
					self.anti_score = self.anti_score + 1
				

			if self.marker >= len(self.frame)-1:
				self.screen.clear()
				self.screen.setText('Finished')
				self.class_lab.setText('GIANT TEST: {}/{}'.format(self.on, self.total_amount))	
				self.screen.repaint()

			else:	
				self.marker += 1
				self.screen.clear()
				if self.frame[self.marker].iloc[2].split('.')[-1] == 'png':
					space = '{}/{}/{}/images/{}'.format(path, self.frame[self.marker].iloc[0], self.frame[self.marker].iloc[1], self.frame[self.marker].iloc[2])

					pixmap = GUI.QPixmap(space)
					pixmap = pixmap.scaled(800, 800, aspectRatioMode=1)
					self.screen.setPixmap(pixmap)
					self.screen.setScaledContents(True)
					self.screen.repaint()

				elif self.frame[self.marker].iloc[2].split('.')[-1] == 'mov' or self.frame[self.marker].iloc[2].split('.')[-1] == 'mp4':
					space = '{}/{}/{}/images/{}'.format(path, self.frame[self.marker].iloc[0], self.frame[self.marker].iloc[1], self.frame[self.marker].iloc[2])
					helper = subprocess.call(['open' , space])

				elif self.frame[self.marker].iloc[2].split('.')[-1] == 'mp3' or self.frame[self.marker].iloc[2].split('.')[-1] == 'MP3':
					space = '{}/{}/{}/images/{}'.format(path, self.frame[self.marker].iloc[0], self.frame[self.marker].iloc[1], self.frame[self.marker].iloc[2])
					playsound.playsound(space)	

				else:	
					self.screen.setText(self.frame[self.marker].iloc[2])
					self.screen.repaint()

				self.on += 1
				self.class_lab.setText('GIANT TEST: {}/{}'.format(self.on, self.total_amount))	
			bottom_side = list(self.frame[self.marker].index)[3]
			self.flip_button.setText(bottom_side)
			self.flip_button.repaint()	

#=========================================
#	              Scoring
#=========================================		

root_path = os.getcwd()
path = os.getcwd()
class score_tab(Wig.QMainWindow):
	def __init__(self, classes, subjects):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 450, 1000
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle('Edit Table') # Window Title
		self.window = Score_tab(classes, subjects)
		self.setCentralWidget(self.window)

class Score_tab(Wig.QWidget):
	def __init__(self, classes, subjects):
		super().__init__()
		vlayout = Wig.QVBoxLayout(self)

		self.class_name = classes
		self.subject_name = subjects

		self.class_wheel = Wig.QLabel(classes)
		self.class_wheel.setAlignment(Qt.Qt.AlignCenter)
		self.subject_wheel = Wig.QLabel(subjects)
		self.subject_wheel.setAlignment(Qt.Qt.AlignCenter)

		self.types = Wig.QLineEdit(self)
		self.search = Wig.QPushButton('Search')
		self.search.clicked.connect(self.search_for)

		self.table = Wig.QTableView()
		self.change_table()
		self.table.clicked.connect(self.fill_base_click)
		self.table.doubleClicked.connect(self.image_pop)

		self.selection_Model = self.table.selectionModel()
		self.selection_Model.selectionChanged.connect(self.fill_base)

		self.edit_line = Wig.QTextEdit(self)
		self.edit_line.setFixedHeight(200)
		self.edit_button = Wig.QPushButton('Edit')

		shortcut = Wig.QShortcut(GUI.QKeySequence("Ctrl+B"),self)
		shortcut.activated.connect(self.bold_text)

		self.edit_button.clicked.connect(self.edit)

		self.remove_button = Wig.QPushButton('Remove (Please select from Row 1)')
		self.remove_button.clicked.connect(self.remove_item)

		vlayout.addWidget(self.class_wheel)
		vlayout.addWidget(self.subject_wheel)
		vlayout.addWidget(self.types)
		vlayout.addWidget(self.search)
		vlayout.addWidget(self.table)
		vlayout.addWidget(self.edit_line)
		vlayout.addWidget(self.edit_button)
		vlayout.addWidget(self.remove_button)

	def bold_text(self):
		self.edit_line.setFontWeight(GUI.QFont.Bold)
		#self.definition.setFontSize(20)
		

	def fill_base(self, selected, deselected):
		# Updated to work with button click
		try:
			for i in selected.indexes():
				row = i.row()
				col = i.column()
				try:
					self.edit_line.setHtml(self.data.iloc[row, col])
				except:
					self.edit_line.setText(self.data.iloc[row, col])	
			self.edit_line.repaint()
		except:
			pass	

	def fill_base_click(self, link):
		try:
			row = link.row()
			col = link.column()
			self.edit_line.setHtml(self.data.iloc[row, col])
			self.edit_line.repaint()
		except:
			pass	

	def change_table(self): 
		try:
			class_name = self.class_wheel.text()
			subject_name = self.subject_wheel.text()

			call_main = data_reader(class_name, [subject_name])
			self.data = pd.read_csv(call_main.flashcards_path[0], index_col=0, header=0)

			
			#self.table = Wig.QTableView()
			self.model = TableModel(self.data)
			self.table.setModel(self.model)
			self.selection_Model = self.table.selectionModel()
			self.selection_Model.selectionChanged.connect(self.fill_base)
			self.table.repaint()

		except:
			self.data = pd.DataFrame({'Blank':[]})
			self.model = TableModel(self.data)
			self.table.setModel(self.model)	
			
			self.selection_Model = self.table.selectionModel()
			self.selection_Model.selectionChanged.connect(self.fill_base)
			self.table.repaint()

	def image_pop(self, link):
		# Should encorporate movies as well
		check = link.data().split('.')
		if check[-1] == 'png':
			link = link.data()
			space = '{}/{}/{}/images/{}'.format(path, self.class_name, self.subject_name, link)
			helper = subprocess.call(['open' , space])
		elif check[-1] == 'mov' or check[-1] == 'mp4':
			link = link.data()
			space = '{}/{}/{}/images/{}'.format(path, self.class_name, self.subject_name, link)
			helper = subprocess.call(['open' , space])
		elif check[-1] == 'MP3' or check[-1] == 'mp3':
			link = link.data()
			space = '{}/{}/{}/images/{}'.format(path, self.class_name, self.subject_name, link)
			playsound.playsound(space)
				

	def edit(self):
		try:
			call_main = data_reader(self.class_name, [self.subject_name])
			row = self.table.currentIndex().row()
			col = self.table.currentIndex().column()

			data = pd.read_csv(call_main.flashcards_path[0], index_col=0, header=0)
			check = data.iloc[row,col].split('.')

			if check[-1] == 'png' or check[-1] == 'mov' or check[-1] == 'mp4' or check[-1] == 'MP3' or check[-1] == 'mp3':
				space_old = '{}/{}/{}/images/{}'.format(path, self.class_name, self.subject_name, data.iloc[row,col])
				space = '{}/{}/{}/images/{}'.format(path, self.class_name, self.subject_name, self.edit_line.toPlainText())
				os.rename(space_old, space)

			
			data.iloc[row,col] = self.edit_line.toHtml()
			data.to_csv(call_main.flashcards_path[0], index=True)

			self.change_table()
			self.edit_line.repaint()
		except:
			pass	

	def remove_item(self):

		indexes = self.table.selectionModel().selectedRows()
		all_indexes = []
		for i in indexes:
			all_indexes += [i.row()]

		# Want to change this to remove when multiple things in the row
		
		#row = self.table.currentIndex().row()
		#col = self.table.currentIndex().column()

		#text = self.table.currentIndex().data()
		
		#category = self.types.currentText()
		call = data_reader(self.class_name, [self.subject_name])

		#if category == 'Flashcards':
		data = pd.read_csv(call.flashcards_path[0], index_col=0, header=0)
		call_it = data.iloc[all_indexes, :]
		check_1 = call_it['Term'].values
		check_2 = call_it['Definition'].values

		#check1 = check_1.split('.')
		#check2 = check_2.split('.')

		for i in check_1:
			check1 = i.split('.')
			if check1[-1] == 'png' or check1[-1] == 'mov' or check1[-1] == 'mp4':
				image1 = path + '/{}/{}/images/{}'.format(self.class_name, self.subject_name, i)
				os.remove(image1)

		for i in check_2:		
			check2 = i.split('.')
			if check2[-1] == 'png' or check2[-1] == 'mov' or check2[-1] == 'mp4':
				image1 = path + '/{}/{}/images/{}'.format(self.class_name, self.subject_name, i)
				os.remove(image1)


		new_data = data.drop(self.data.index[all_indexes])	
			
		new_data = new_data.reset_index(drop=True)
		new_data.to_csv(call.flashcards_path[0], index=True)

		self.change_table()	

	def search_for(self):
		try:
			class_name = self.class_wheel.text()
			subject_name = self.subject_wheel.text()

			call_main = data_reader(class_name, [subject_name])
			self.data = pd.read_csv(call_main.flashcards_path[0], index_col=0, header=0)

			
			#self.table = Wig.QTableView()
			self.model = SearchModel(self.data, self.types.text(), class_name, subject_name)
			self.table.setModel(self.model)
			self.selection_Model = self.table.selectionModel()
			self.selection_Model.selectionChanged.connect(self.fill_base)
			self.table.repaint()

		except:
			pass
		#	self.data = pd.DataFrame({'Blank':[]})
		#	self.model = TableModel(self.data)
		#	self.table.setModel(self.model)	
			
		#	self.selection_Model = self.table.selectionModel()
		#	self.selection_Model.selectionChanged.connect(self.fill_base)
		#	self.table.repaint()
	
#=========================================
#	            Home Tab
#=========================================
class Home(Wig.QWidget):
	# This doesnt seem to work for Shortmer information
	def __init__(self, parent):
		super(Home, self).__init__(parent)

		back_path = os.getcwd().split('/')[:-1]
		self.back_path = '/'.join(back_path)


		grid = Wig.QGridLayout()
		grid_width = 700

		self.switch = Wig.QComboBox(self)
		#self.switch.addItem('Today')     # Removing Today section
		self.switch.addItem('Learning')
		self.switch.addItem('Forgetting Test Score')
		self.switch.addItem('Forgetting Estimated Score')
		self.switch.addItem('Memorized')
		self.switch.addItem('Everything')

		self.sc = Canvas(self, width=5, height=3)
		self.sc.setMinimumSize(100,150)

		self.refresh_button = Wig.QPushButton('Refresh')
		shortcut = Wig.QShortcut(GUI.QKeySequence("Shift+Return"),self)
		shortcut.activated.connect(self.ow_my)
		self.refresh_button.clicked.connect(self.ow_my)

		self.refresh_button.setFixedWidth(grid_width)

		self.switch.currentTextChanged.connect(self.change_display)
		self.switch.setFixedWidth(grid_width)

		self.large_list  = Wig.QListWidget(self)
		self.large_list.setFixedWidth(grid_width)


		self.how_many = Wig.QLabel()
		self.how_many.setAlignment(Qt.Qt.AlignCenter)
		self.how_many.setFixedWidth(grid_width)

		# Need to check all the classes and subjects to see if they
		# qualify for the home board

		# Initiate Home Screen as Learning Screen

		
		try:
			today_listed = home_reader2().learning()
			today_listed = today_listed.sort_values(by=['C'])
			today_listed = today_listed.reset_index(drop=True)
			self.setting = access_settings(R'settings.csv', change = 0)
			#goals = int(self.setting['Goals'].values[0])
			#tracker = int(self.setting['tracker'].values[0])
			font = int(self.setting['text_size'].values[0])

			self.large_list.setStyleSheet('font-size: {}px'.format(font))

			#goals = goals - tracker
		
			#today_listed = today_listed.sort_values(by=['C'])
			#today_listed = today_listed.iloc[:goals]

			self.ones = len(today_listed)
			self.how_many.setText(str(self.ones))

			for i, w in today_listed.iterrows():
					
				self.large_list.addItem('{} -- {} -- {:.2f}'.format(w['A'], w['B'], w['C']))

				try:
					self.focus = pd.read_csv(self.back_path + '/focus.csv', index_col=0, header=0)
					for j in self.focus.values:
						
						if w['A'] == j[0] and w['B'] == j[1]:
							if j[2] == 'Green':
								self.large_list.item(i).setBackground(Qt.Qt.green)

							elif j[2] == 'Blue':
								self.large_list.item(i).setBackground(Qt.Qt.darkBlue)
								self.large_list.item(i).setForeground(Qt.Qt.white)

							elif j[2] == 'Yellow':
								self.large_list.item(i).setBackground(Qt.Qt.yellow)
						
							elif j[2] == 'Dark Green':
								self.large_list.item(i).setBackground(Qt.Qt.darkGreen)
							else:
								self.large_list.item(i).setBackground(Qt.Qt.black)
								self.large_list.item(i).setForeground(Qt.Qt.white)
					
				except:
					pass			

		except:
			pass



		self.large_list.setFixedWidth(grid_width)	

		try:	
			self.ones = len(learning_listed)
		except:
			self.ones = 0	

		self.large_list.currentItemChanged.connect(self.roll_change_map)		

		self.limit = Wig.QComboBox(self)
		self.limit.setFixedWidth(grid_width)

		try:
			classes = pd.read_csv(self.back_path + '/class_map.csv', header=0, index_col=0)
			classes = classes['Class'].values

		
			self.limit.addItem('All')
			for i in classes:
				self.limit.addItem(i)
		except:
			pass		
		self.limit.currentTextChanged.connect(self.change_display)

		#----------------------------------------------------------

		grid.addWidget(self.refresh_button,0,0)
		grid.addWidget(self.switch,1,0)
		grid.addWidget(self.how_many,2,0)
		grid.addWidget(self.large_list,3,0)
		grid.addWidget(self.limit,4,0)

		grid_all = Wig.QGridLayout()

		quick_test = Wig.QPushButton('Take Test')
		quick_test.clicked.connect(self.test_path2)

		self.Notes2 = Wig.QLabel()
		self.Notes2.setFixedHeight(45)

		self.s_select = Wig.QDoubleSpinBox(self, decimals=3, maximum=10000, minimum=0.0001)
		self.s_select.valueChanged.connect(self.day_calc)

		self.s_select.setAlignment(Qt.Qt.AlignCenter)
		self.s_select.setFixedWidth(100)
		self.s_select.setPrefix('s: ')

		enter = Wig.QShortcut(GUI.QKeySequence(Qt.Qt.Key_Return), self)
		enter.activated.connect(self.alter_s)

		# Custom BUttons
		#----------------------------------------------
		hori = Wig.QHBoxLayout()
		vert = Wig.QVBoxLayout()
		
		quick_add  = PicButton(self.back_path+'/button/add.png', self.back_path+'/button/add_hov.png', self.back_path+'/button/add_pressed.png',self)
		quick_edit = PicButton(self.back_path+'/button/edit.png', self.back_path+'/button/edit_hov.png', self.back_path+'/button/edit_pressed.png', self)
		Forgetting = PicButton(self.back_path+'/button/forgetting.png', self.back_path+'/button/forgetting_hov.png', self.back_path+'/button/forgetting_pressed.png', self)
		learning   = PicButton(self.back_path+'/button/learning.png', self.back_path+'/button/learning_hov.png', self.back_path+'/button/learning_pressed.png' ,self)
		Memorized  = PicButton(self.back_path+'/button/memorized.png', self.back_path+'/button/mem_hov.png', self.back_path+'/button/mem_pressed.png', self)

		learning.clicked.connect(self.roll_over_learn)
		Forgetting.clicked.connect(self.roll_over_forg)
		Memorized.clicked.connect(self.roll_over_mem)

		quick_add.clicked.connect(self.add_path2)
		quick_edit.clicked.connect(self.edit_path2)
	   
		vert.addStretch(0)
		vert.addWidget(quick_add)
		vert.addWidget(Memorized)
		vert.addWidget(quick_edit)
		vert.addStretch(0)

		hori.addStretch(0)
		hori.addWidget(learning)
		hori.addLayout(vert)
		hori.addWidget(Forgetting)
		hori.addStretch(0)

		grid_all.addWidget(quick_test,0,0)
		grid_all.addWidget(self.sc,1,0)

		grid_all.addWidget(self.Notes2,2,0)
		grid_all.setRowStretch(3,1)
		grid_all.addWidget(self.s_select,3,0, alignment=Qt.Qt.AlignCenter)
		grid_all.addLayout(hori,4,0)
		#---------------------------------------------------------

		all_grid = Wig.QGridLayout(self)
		all_grid.addLayout(grid, 0,0)
		all_grid.addLayout(grid_all, 0,1)

		self.large_list.repaint()

	def roll_change_map(self):
		self.sc.axes.clear()
		try:
			link = self.large_list.currentItem().text()
			link = link.split(' -- ')
			#print(link)
			self.classes = link[0]
			self.subject = link[1]

			call = data_reader(self.classes, [self.subject])
		
			curve = call.plot_data2()
			self.sc.axes.plot(curve[0], curve[1])
			self.sc.draw()

			try:		
		
				gen_data = pd.read_csv(call.general_data_path[0], header=0, index_col=0)

				self.L_F_tell = gen_data.iloc[-1,5]
				self.j = gen_data.iloc[-1,6]
				self.F = gen_data.iloc[-1,2]

				self.s = gen_data.iloc[-1,4]

				self.days = call.determine_days2(self.s)
				

				self.Notes2.setText('Last Score: {:.2f}\ndays: {:.2f}'.format(self.F, self.days))
				self.Notes2.setAlignment(Qt.Qt.AlignCenter)

				self.s_select.setValue(self.s)
				
			except:
				pass		
			
		except:
			pass


	def day_calc(self, new_value):
		# Cant have both of these it seems.
		if self.L_F_tell == 'F':

			call = data_reader(self.classes, [self.subject])
			self.days = call.determine_days2(new_value)


			self.Notes2.setText('Last Score: {:.2f}\ndays: {:.2f}'.format(self.F,self.days))
			self.Notes2.setAlignment(Qt.Qt.AlignCenter)

		else:
			days = 0


	def alter_s(self):
		# works want it to not be able to change learning but not super importatnt
		try:
			call = data_reader(self.classes, [self.subject])
			gen_data = pd.read_csv(call.general_data_path[0], header=0, index_col=0)
			gen_data.iloc[-1,4] = float(self.s_select.value())
			
			gen_data.to_csv(call.general_data_path[0], header=True, index=True)
		except:
			pass	

# Install warning windows here if have not selected values and click these buttons
	def roll_over_learn(self):
		try:
			call = data_writer(self.classes, self.subject)
			call.add_L_F('L')
			self.ow_my()
			#self.close()
		except:
			pass	

	def roll_over_forg(self):
		try:
			call = data_writer(self.classes, self.subject)
			call.add_L_F('F')
			#self.close()
		except:
			pass	

	def roll_over_mem(self):
		try:
			call = data_writer(self.classes, self.subject)
			call.add_L_F('M')
			#self.close()
		except:
			pass	

	def test_path2(self):
		self.day_check = self.switch.currentText()
		try:
		# Can perform quick tests now need this to limit the today tab alittle
		# better

		# Fix this here!!
		
			if self.day_check == 'Today':
				setting = access_settings(R'settings.csv', change = 0)
				self.goals = setting['Goals'].values[0]
				self.so_far = setting['tracker'].values[0]
				self.maintain = setting['maintain'].values[0]
				self.path = setting['text_size'].values[0]

				self.so_far = self.so_far + 1
				self.date = setting['date'].values[0]

				setting = access_settings(R'settings.csv', change = [self.goals, self.so_far, self.date, self.maintain, self.path])

			else:
				pass
		except:
			pass			


		call = data_writer(self.classes, self.subject)
		#call.add_study_time('0')	OUTDATED amount of study time isnt added anymore
		self.w = test_screen_pop(self.classes, self.subject)
		self.w.show()

		# Work more on this path later, It needs to send a value if this test is from the today catgory

	def add_path2(self):
		self.call = adding_material(self.classes, self.subject)
		self.call.show()

	def edit_path2(self):
		self.call = score_tab(self.classes, self.subject)
		self.call.show()

	def change_display(self):
		
		self.large_list.clear()
		maintain = int(self.setting['maintain'].values[0])
		
		if self.switch.currentText() == 'Learning':
			try:
				learning_listed = home_reader2().learning()
				learning_listed = learning_listed.sort_values(by=['C'])
				learning_listed = learning_listed.reset_index(drop=True)
				

				if self.limit.currentText() == 'All':
					pass
				else:
					learning_listed = learning_listed[learning_listed['A'] == self.limit.currentText()]	
					learning_listed = learning_listed.reset_index(drop=True)

				self.ones = len(learning_listed)
				self.how_many.setText(str(self.ones))

				for i, w in learning_listed.iterrows():
					
					self.large_list.addItem('{} -- {} -- {:.2f}'.format(w['A'], w['B'], w['C']))

					try:
						self.focus = pd.read_csv(self.back_path + '/focus.csv', index_col=0, header=0)
						for j in self.focus.values:
							if w['A'] == j[0] and w['B'] == j[1]:
								if j[2] == 'Green':
									self.large_list.item(i).setBackground(Qt.Qt.green)

								elif j[2] == 'Blue':
									self.large_list.item(i).setBackground(Qt.Qt.darkBlue)
									self.large_list.item(i).setForeground(Qt.Qt.white)

								elif j[2] == 'Yellow':
									self.large_list.item(i).setBackground(Qt.Qt.yellow)
								
								elif j[2] == 'Dark Green':
									self.large_list.item(i).setBackground(Qt.Qt.darkGreen)
								else:
									self.large_list.item(i).setBackground(Qt.Qt.black)
									self.large_list.item(i).setForeground(Qt.Qt.white)
											
										
					except:
						pass			
			
			except:
				pass

		if self.switch.currentText() == 'Everything':
			try:
				learning_listed = home_reader2().everything()
				learning_listed = learning_listed.sort_values(by=['C'])
				learning_listed = learning_listed.reset_index(drop=True)
				if self.limit.currentText() == 'All':
					pass
				else:
					learning_listed = learning_listed[learning_listed['A'] == self.limit.currentText()]
					learning_listed = learning_listed.reset_index(drop=True)

				self.ones = len(learning_listed)
				self.how_many.setText(str(self.ones))

				for i, w in learning_listed.iterrows():
				
					self.large_list.addItem('{} -- {} -- {:.2f}'.format(w['A'], w['B'], w['C']))

					# Error thrown up here
					if float(w['C']) <= int(maintain)*0.01:
						self.large_list.item(i).setForeground(Qt.Qt.red)

					try:
						self.focus = pd.read_csv(self.back_path + '/focus.csv', index_col=0, header=0)
						for j in self.focus.values:
							if w['A'] == j[0] and w['B'] == j[1]:
								if j[2] == 'Green':
									self.large_list.item(i).setBackground(Qt.Qt.green)
								elif j[2] == 'Blue':
									self.large_list.item(i).setBackground(Qt.Qt.darkBlue)
									self.large_list.item(i).setForeground(Qt.Qt.white)
								
								elif j[2] == 'Yellow':
									self.large_list.item(i).setBackground(Qt.Qt.yellow)
					
								elif j[2] == 'Dark Green':
									self.large_list.item(i).setBackground(Qt.Qt.darkGreen)
								else:
									self.large_list.item(i).setBackground(Qt.Qt.black)
									self.large_list.item(i).setForeground(Qt.Qt.white)		
										
					except:
						pass
			
			except:
				pass		

		elif self.switch.currentText() == 'Today':
			
			try:
				# Still need to make faster
				today_listed = home_reader2().today()
				setting = access_settings(R'settings.csv', change = 0)
				goals = int(setting['Goals'].values[0])
				tracker = int(setting['tracker'].values[0])

				goals = goals - tracker

				if self.limit.currentText() == 'All':
					pass
				else:
					today_listed = today_listed[today_listed['A'] == self.limit.currentText()]
					today_listed = today_listed.reset_index(drop=True)
	
				today_listed = today_listed.sort_values(by=['C'])
				today_listed = today_listed.reset_index(drop=True)
				today_listed = today_listed.iloc[:goals]

				self.ones = len(today_listed)
				self.how_many.setText(str(self.ones))

				for i, w in today_listed.iterrows():
					self.large_list.addItem('{} -- {} -- {:.2f}'.format(w['A'], w['B'], w['C']))

					try:
						self.focus = pd.read_csv(self.back_path + '/focus.csv', index_col=0, header=0)
						for j in self.focus.values:
							if w['A'] == j[0] and w['B'] == j[1]:
								if j[2] == 'Green':
									self.large_list.item(i).setBackground(Qt.Qt.green)
								elif j[2] == 'Blue':
									self.large_list.item(i).setBackground(Qt.Qt.darkBlue)
									self.large_list.item(i).setForeground(Qt.Qt.white)
								
								elif j[2] == 'Yellow':
									self.large_list.item(i).setBackground(Qt.Qt.yellow)

								elif j[2] == 'Dark Green':
									self.large_list.item(i).setBackground(Qt.Qt.darkGreen)
								else:
									self.large_list.item(i).setBackground(Qt.Qt.black)
									self.large_list.item(i).setForeground(Qt.Qt.white)
											
											
					except:
						pass

			except:
				pass

		elif self.switch.currentText() == 'Forgetting Test Score':
			
			try:
				# Still need to make faster
				forgetting_listed = home_reader2().forgetting_test_score()
				forgetting_listed = forgetting_listed.sort_values(by=['C'])
				forgetting_listed = forgetting_listed.reset_index(drop=True)

				if self.limit.currentText() == 'All':
					pass
				else:
					forgetting_listed = forgetting_listed[forgetting_listed['A'] == self.limit.currentText()]
					forgetting_listed = forgetting_listed.reset_index(drop=True)	

				self.ones = len(forgetting_listed)
				self.how_many.setText(str(self.ones))

				for i, w in forgetting_listed.iterrows():
					self.large_list.addItem('{} -- {} -- {:.2f}'.format(w['A'], w['B'], w['C']))

					if float(w['C']) <= int(maintain)*0.01:
						self.large_list.item(i).setForeground(Qt.Qt.red)

					try:
						self.focus = pd.read_csv(self.back_path + '/focus.csv', index_col=0, header=0)
						for j in self.focus.values:
							if w['A'] == j[0] and w['B'] == j[1]:
								if j[2] == 'Green':
									self.large_list.item(i).setBackground(Qt.Qt.green)
								elif j[2] == 'Blue':
									self.large_list.item(i).setBackground(Qt.Qt.darkBlue)
									self.large_list.item(i).setForeground(Qt.Qt.white)
								
								elif j[2] == 'Yellow':
									self.large_list.item(i).setBackground(Qt.Qt.yellow)

								elif j[2] == 'Dark Green':
									self.large_list.item(i).setBackground(Qt.Qt.darkGreen)
								else:
									self.large_list.item(i).setBackground(Qt.Qt.black)
									self.large_list.item(i).setForeground(Qt.Qt.white)
											
										
					except:
						pass
			
			except:
				pass		

		elif self.switch.currentText() == 'Forgetting Estimated Score':
			# Still need to make faster
			try:
				forgetting_listed = home_reader2().today()
				forgetting_listed = forgetting_listed.sort_values(by=['C'])
				forgetting_listed = forgetting_listed.reset_index(drop=True)

				if self.limit.currentText() == 'All':
					pass
				else:
					forgetting_listed = forgetting_listed[forgetting_listed['A'] == self.limit.currentText()]
					forgetting_listed = forgetting_listed.reset_index(drop=True)

				self.ones = len(forgetting_listed)
				self.how_many.setText(str(self.ones))

				for i, w in forgetting_listed.iterrows():
					self.large_list.addItem('{} -- {} -- {:.2f}'.format(w['A'], w['B'], w['C']))

					if float(w['C']) <= int(maintain)*0.01:
						self.large_list.item(i).setForeground(Qt.Qt.red)
					try:
						self.focus = pd.read_csv(self.back_path + '/focus.csv', index_col=0, header=0)
						for j in self.focus.values:
							if w['A'] == j[0] and w['B'] == j[1]:
								if j[2] == 'Green':
									self.large_list.item(i).setBackground(Qt.Qt.green)
								elif j[2] == 'Blue':
									self.large_list.item(i).setBackground(Qt.Qt.darkBlue)
									self.large_list.item(i).setForeground(Qt.Qt.white)
								
								elif j[2] == 'Yellow':
									self.large_list.item(i).setBackground(Qt.Qt.yellow)

								elif j[2] == 'Dark Green':
									self.large_list.item(i).setBackground(Qt.Qt.darkGreen)
								else:
									self.large_list.item(i).setBackground(Qt.Qt.black)
									self.large_list.item(i).setForeground(Qt.Qt.white)
											
										
					except:
						pass
			
			except:
				pass
		
		elif self.switch.currentText() == 'Memorized':
			try:
				# Still need to make faster
				forgetting_listed = home_reader2().memorized()
				forgetting_listed = forgetting_listed.sort_values(by=['C'])
				forgetting_listed = forgetting_listed.reset_index(drop=True)

				if self.limit.currentText() == 'All':
					pass
				else:
					forgetting_listed = forgetting_listed[forgetting_listed['A'] == self.limit.currentText()]
					forgetting_listed = forgetting_listed.reset_index(drop=True)

				self.ones = len(forgetting_listed)
				self.how_many.setText(str(self.ones))

				for i, w in forgetting_listed.iterrows():
					self.large_list.addItem('{} -- {} -- {:.2f}'.format(w['A'], w['B'], w['C']))

					try:
						self.focus = pd.read_csv(self.back_path + '/focus.csv', index_col=0, header=0)
						for j in self.focus.values:
							if w['A'] == j[0] and w['B'] == j[1]:
								if j[2] == 'Green':
									self.large_list.item(i).setBackground(Qt.Qt.green)
								elif j[2] == 'Blue':
									self.large_list.item(i).setBackground(Qt.Qt.darkBlue)
									self.large_list.item(i).setForeground(Qt.Qt.white)
								
								elif j[2] == 'Yellow':
									self.large_list.item(i).setBackground(Qt.Qt.yellow)

								elif j[2] == 'Dark Green':
									self.large_list.item(i).setBackground(Qt.Qt.darkGreen)
								else:
									self.large_list.item(i).setBackground(Qt.Qt.black)
									self.large_list.item(i).setForeground(Qt.Qt.white)
											
										
					except:
						pass

			except:
				pass

		self.large_list.repaint()

	def move_pop(self, link):
		#try:
		link = link.data()
		self.w = move_show(link, self.switch.currentText())
		self.w.show()
		#except:
		#	pass	

	def ow_my(self):
		#try:	
		self.limit.clear()
		classes = pd.read_csv(self.back_path + '/class_map.csv', header=0, index_col=0)
		classes = classes['Class'].values

		
		self.limit.addItem('All')
		for i in classes:
			self.limit.addItem(i)
		#except:
		#	pass


		self.change_display()	

#=========================================
#	        Changing Settings
#=========================================		
class changing_sett(Wig.QMainWindow):
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 300, 300
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Change Settings") # Window Title
		self.window = changing_setting_window()
		self.setCentralWidget(self.window)


class changing_setting_window(Wig.QWidget):
	def __init__(self):
		super().__init__()

		frame = access_settings(R'settings.csv', change = 0)
		goals = int(frame.values[0][0])
		self.date  = frame.values[0][2]
		grid = Wig.QGridLayout(self)

		goal_label = Wig.QLabel('Goals')
		self.score_label = Wig.QLineEdit(self)
		self.score_label.setText(str(goals))
		self.score_label.setAlignment(Qt.Qt.AlignCenter)

		percent_label = Wig.QLabel('Tracker')
		self.warn_label = Wig.QLineEdit(self)
		self.warn_label.setText(str(frame.values[0][1]))
		self.warn_label.setAlignment(Qt.Qt.AlignCenter)

		maintain_label = Wig.QLabel('Maintenance Level')
		self.maintenance = Wig.QLineEdit(self)
		self.maintenance.setText(str(frame.values[0][3]))
		self.maintenance.setAlignment(Qt.Qt.AlignCenter)
		percentage_label = Wig.QLabel('%')

		file_package_path = Wig.QLabel('Text Size')
		self.file_path = Wig.QLineEdit(self)
		self.file_path.setText(str(frame.values[0][4]))
		self.file_path.setAlignment(Qt.Qt.AlignCenter)

		go_button = Wig.QPushButton('Apply Changes')
		go_button.clicked.connect(self.apply_changes)

		grid.addWidget(goal_label,0,0)
		grid.addWidget(self.score_label,0,1)
		grid.addWidget(percent_label,1,0)
		grid.addWidget(self.warn_label,1,1)
		grid.addWidget(maintain_label,2,0)
		grid.addWidget(self.maintenance,2,1)
		grid.addWidget(percentage_label,2,2)
		grid.addWidget(file_package_path,3,0)
		grid.addWidget(self.file_path,3,1)
		grid.addWidget(go_button,4,1)

	def apply_changes(self):
		new_value_goals = int(self.score_label.text())
		new_value_per = int(self.warn_label.text())
		new_value_main = int(self.maintenance.text())
		new_file_path = str(self.file_path.text())
		access_settings(R'settings.csv', change = [new_value_goals, new_value_per, self.date, new_value_main, new_file_path])	

#=========================================
#	           Arrangments
#=========================================	
# Before I do this I need to readjust the imaging portions of this 
# projects

class arrange_nav_files(Wig.QMainWindow):
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 1000, 500
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Arrange Files") # Window Title
		self.window = arrange1()
		self.setCentralWidget(self.window)

class arrange1(Wig.QWidget):
	def __init__(self):
		super().__init__()

		width1 = 150
		width2 = 850

		grid1 = Wig.QGridLayout()
		grid2 = Wig.QGridLayout()
		grid3 = Wig.QGridLayout()
		grid4 = Wig.QGridLayout(self)

		label = Wig.QLabel('New Class')
		label.setFixedWidth(width1)
		label.setAlignment(Qt.Qt.AlignCenter)

		self.title_box = Wig.QLineEdit(self)
		self.title_box.setFixedWidth(width2)
		self.add_folder= Wig.QPushButton('Add Subject')
		self.add_folder.clicked.connect(self.find_folder)
		self.add_folder.setFixedWidth(width1)

		self.remove_folder = Wig.QPushButton('Remove Folder')
		self.remove_folder.setFixedWidth(width1)

		self.remove_folder.clicked.connect(self.removing)
		
		self.lister = Wig.QListWidget(self)
		self.lister.setFixedWidth(width2)

		activate = Wig.QPushButton('Make New Class out of Subjects')
		activate.clicked.connect(self.activate_fusion)

		grid1.addWidget(label,0,0)
		grid1.addWidget(self.title_box,0,1)
		
		grid2.addWidget(self.add_folder,0,0)
		grid2.addWidget(self.remove_folder,1,0)
		grid2.setRowStretch(3,1)

		grid3.addLayout(grid2,0,0)
		grid3.addWidget(self.lister,0,1)

		grid4.addLayout(grid1,0,0)
		grid4.addLayout(grid3,1,0)
		grid4.addWidget(activate,2,0)


	def find_folder(self):
		path = det_path(R'Data')
		file = Wig.QFileDialog.getExistingDirectory(self, 'Open Folder', path)
		self.lister.addItem(str(file))

	def find_file(self):
		path = det_path(R'Data')
		file = Wig.QFileDialog.getOpenFileName(self, 'Open File', path) 
		self.lister.addItem(str(file[0]))

	def removing(self):
		self.lister.takeItem(self.lister.currentRow())
		self.lister.repaint()	

	def activate_fusion(self):
		new_class = self.title_box.text()
		new_class_path = '{}/{}'.format(os.getcwd(), new_class)
		os.mkdir(new_class_path)

		amount = self.lister.count()

		for i in range(amount):
			piece = self.lister.item(i).text()
			piece2 = piece.split('/')[-2:]
			piece2 = ' - '.join(piece2)

			local = new_class_path + '/' + piece2
			shutil.move(piece, local)

		self.title_box.clear()
		self.lister.clear()	

#-----------------------------------------------------------------
class combine_files_func(Wig.QMainWindow):
	# This has been taken out for now needs a complete overhaul in the
	# future

	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 1000, 500
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Arrange Files") # Window Title
		self.window = arrange2()
		self.setCentralWidget(self.window)

class arrange2(Wig.QWidget):
	def __init__(self):
		super().__init__()

		width1 = 150
		width2 = 850

		grid1 = Wig.QGridLayout()
		grid2 = Wig.QGridLayout()
		grid3 = Wig.QGridLayout()
		grid4 = Wig.QGridLayout(self)

		class_setup_label = Wig.QLabel('Select Class')
		class_setup_label.setFixedWidth(width1)
		class_setup_label.setAlignment(Qt.Qt.AlignCenter)
		self.class_setup = Wig.QComboBox(self)
		classes = list(os.walk(os.getcwd()))[0][1]
		classes = np.sort(classes)
		
		for i in classes:
			self.class_setup.addItem(i)

		self.class_setup.setFixedWidth(width2)	

		label = Wig.QLabel('New Subject')
		label.setFixedWidth(width1)
		label.setAlignment(Qt.Qt.AlignCenter)

		self.title_box = Wig.QLineEdit(self)
		self.title_box.setFixedWidth(width2)
		self.add_folder= Wig.QPushButton('Add Subject')
		self.add_folder.clicked.connect(self.find_folder)
		self.add_folder.setFixedWidth(width1)

		self.remove_folder = Wig.QPushButton('Remove Subject')
		self.remove_folder.setFixedWidth(width1)

		self.remove_folder.clicked.connect(self.removing)
		
		self.lister = Wig.QListWidget(self)
		self.lister.setFixedWidth(width2)

		activate = Wig.QPushButton('Make New Subject out of Subjects')
		activate.clicked.connect(self.activate_fusion)

		grid1.addWidget(class_setup_label,0,0)
		grid1.addWidget(self.class_setup,0,1)
		grid1.addWidget(label,1,0)
		grid1.addWidget(self.title_box,1,1)
		
		grid2.addWidget(self.add_folder,0,0)
		grid2.addWidget(self.remove_folder,1,0)
		grid2.setRowStretch(3,1)

		grid3.addLayout(grid2,0,0)
		grid3.addWidget(self.lister,0,1)


		grid4.addLayout(grid1,0,0)
		grid4.addLayout(grid3,1,0)
		grid4.addWidget(activate,2,0)


	def find_folder(self):
		path = det_path(R'Data')
		file = Wig.QFileDialog.getExistingDirectory(self, 'Open Folder', path)
		self.lister.addItem(str(file))

	def find_file(self):
		path = det_path(R'Data')
		file = Wig.QFileDialog.getOpenFileName(self, 'Open File', path) 
		self.lister.addItem(str(file[0]))

	def removing(self):
		self.lister.takeItem(self.lister.currentRow())
		self.lister.repaint()	

	def activate_fusion(self):
		new_class = self.class_setup.currentText()
		new_subject = self.title_box.text()
		new_class_path = '{}/{}/{}'.format(os.getcwd(), new_class, new_subject)
		os.mkdir(new_class_path)
		new_class_path_images = new_class_path + '/images'
		os.mkdir(new_class_path_images)

		amount = self.lister.count()

		diagrams = pd.DataFrame()
		drawings = pd.DataFrame()
		flashcards = pd.DataFrame()
		practice = pd.DataFrame()
		images = []

		for i in range(amount):
			piece = self.lister.item(i).text()
			new_diagrams = piece + '/diagrams.csv'
			new_drawings = piece + '/drawings.csv'
			new_flash = piece + '/flashcards.csv'
			new_pract = piece + '/practice_qs.csv'
			new_images = piece + '/images'

			diag = pd.read_csv(new_diagrams, index_col=0, header=0)
			draw = pd.read_csv(new_drawings, index_col=0, header=0)
			flash= pd.read_csv(new_flash, index_col=0, header=0)
			pract= pd.read_csv(new_pract, index_col=0, header=0)

			image = list(os.walk(new_images))[0][2]
			if len(image) > 1:
				for i in image:
					shutil.move(new_images + '/{}'.format(i), new_class_path_images + '/{}'.format(i))
			else:
				pass		

			diagrams = diagrams.append(diag, ignore_index=True)
			drawings = drawings.append(draw, ignore_index=True)
			flashcards = flashcards.append(flash, ignore_index=True)
			practice = practice.append(pract, ignore_index=True)

			shutil.rmtree(piece)

		diagrams.to_csv(new_class_path + '/diagrams.csv', index=True)
		drawings.to_csv(new_class_path + '/drawings.csv', index=True)
		flashcards.to_csv(new_class_path + '/flashcards.csv', index=True)
		practice.to_csv(new_class_path + '/practice_qs.csv', index=True)

		general_data = pd.DataFrame({'Subjects':[new_subject],
									 'Date':[datetime.datetime.now()],
									 'F':[0],
									 'base':[0],
									 's_values':[0.14],
									 'L_F':['L'],
									 'j':[0],
									 'studied (min)':[0]})
		general_data.to_csv(new_class_path + '/general_data.csv', index=True)

		mach_data = open(new_class_path + '/mach_data.csv', 'w+')

		data = pd.Series([0])
		# Need to change here
		data.to_csv(new_class_path + '/curves_data.csv', index=False)

		self.title_box.clear()
		self.lister.clear()		

#=========================================
#	           Adding
#=========================================	

path = os.getcwd()

class ADD_FILE(Wig.QMainWindow):
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 300, 200
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Add Class") # Window Title
		self.window = adder1()
		self.setCentralWidget(self.window)


class adder1(Wig.QWidget):
	def __init__(self):
		super().__init__()

		grid = Wig.QVBoxLayout(self)

		label1 = Wig.QLabel('Select Class to add File to')
		label1.setFixedHeight(10)
		label1.setAlignment(Qt.Qt.AlignCenter)	
		self.class_wheel = Wig.QComboBox(self)
		try:
			walker = list(os.walk(os.getcwd()))
			classes = walker[0][1]
			classes = np.sort(classes)
			for i in classes:
				self.class_wheel.addItem(i)	
		except:
			pass

		
		self.check = Wig.QCheckBox('Automatically place in Focus')
		self.check.setCheckState(Qt.Qt.Checked)	

		self.colors = Wig.QComboBox(self)
		cols = ['Green', 'Blue', 'Yellow', 'Gray', 'Dark Green', 'Black']
		back_path = path.split('/')
		back_path = back_path[:-1]
		self.back_path = '/'.join(back_path)

		try:
			old_data = pd.read_csv(self.back_path+'/focus.csv', index_col=0, header=0)
			last = old_data.iloc[-1,2]
			self.colors.addItem(last)

		except:
			pass	

		
		for i in cols:
			self.colors.addItem(i)

		label2 = Wig.QLabel('Enter File Name')
		label2.setFixedHeight(10)
		label2.setAlignment(Qt.Qt.AlignCenter)

		self.input_line = Wig.QLineEdit(self)
		enter = Wig.QPushButton('Enter')
		enter.clicked.connect(self.add_files_now)

		shortcut = Wig.QShortcut(GUI.QKeySequence("Return"),self)
		shortcut.activated.connect(self.add_files_now)

		grid.addWidget(label1)
		grid.addWidget(self.class_wheel)
		grid.addStretch(0)
		grid.addWidget(self.check)
		grid.addWidget(self.colors)
		grid.addStretch(0)
		grid.addWidget(label2)
		grid.addWidget(self.input_line)
		grid.addWidget(enter)
		grid.addStretch(0)

	def add_files_now(self):
		back_path = path.split('/')
		back_path = back_path[:-1]
		self.back_path = '/'.join(back_path)

		new_path = os.getcwd() + '/{}/{}'.format(self.class_wheel.currentText(), self.input_line.text())
		os.mkdir(new_path)
		os.chdir(new_path)

		general_data = pd.DataFrame({'Subjects':[self.input_line.text()],
									 'Date':[datetime.datetime.now()],
									 'F':[0],
									 'base':[0],
									 's_values':[0.14],
									 'L_F':['L'],
									 'j':[0],
									 'studied (min)':[0]})
		general_data.to_csv('general_data.csv', index=True)
		

		flashcards = pd.DataFrame(columns=['Term',
										   'Definition',
										   'Correct',
										   'Incorrect'])
		flashcards.to_csv('flashcards.csv',index=True)

		os.mkdir('images')
		
		os.chdir(path)

		if self.check.isChecked():
			new_data = pd.DataFrame({'Class':[self.class_wheel.currentText()],'Subject':[self.input_line.text()], 'Color':[self.colors.currentText()]})
			try:
				old_data = pd.read_csv(self.back_path+'/focus.csv', index_col=0, header=0)

				old_data = old_data.append(new_data, ignore_index=True)
				old_data.to_csv(self.back_path+'/focus.csv', index=True)
			except:
				pass	
		else:
			pass	

		try:
			data_map = pd.read_csv(self.back_path + '/data_map.csv', header=0, index_col=0)
		except:
			data_map = pd.DataFrame({'Class':[], 'Subject':[], 'L_F':[]})
			
		data_map = data_map.append({'Class':self.class_wheel.currentText(), 'Subject':self.input_line.text(), 'L_F':'L'}, ignore_index=True)
		data_map.to_csv(self.back_path + '/data_map.csv', header=True, index=True)	
			

		self.input_line.clear()	


#------------------------------------------------------------

class ADD_CLASS(Wig.QMainWindow):
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 300, 100
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Add Class") # Window Title
		self.window = adder2()
		self.setCentralWidget(self.window)


class adder2(Wig.QWidget):
	def __init__(self):
		super().__init__()

		grid = Wig.QVBoxLayout(self)

		label = Wig.QLabel('Enter New Class')
		label.setFixedHeight(10)
		label.setAlignment(Qt.Qt.AlignCenter)
		self.input_line = Wig.QLineEdit(self)
		enter = Wig.QPushButton('Enter')
		enter.clicked.connect(self.add_class_now)

		shortcut = Wig.QShortcut(GUI.QKeySequence("Return"),self)
		shortcut.activated.connect(self.add_class_now)

		grid.addWidget(label)
		grid.addWidget(self.input_line)
		grid.addWidget(enter)
		grid.addStretch(0)

	def add_class_now(self):
		back_path = path.split('/')
		back_path = back_path[:-1]
		back_path = '/'.join(back_path)

		new_path = os.getcwd() + '/{}'.format(self.input_line.text())
		os.mkdir(new_path)
		

		try:
			class_map = pd.read_csv(back_path + '/class_map.csv', header=0, index_col=0)
		except:
			class_map = pd.DataFrame({'Class':[]})
			
		class_map = class_map.append({'Class':self.input_line.text()}, ignore_index=True)
		class_map.to_csv(back_path + '/class_map.csv', header=True, index=True)
		self.input_line.clear()		
#------------------------------------------------------------
class EDIT_CLASS(Wig.QMainWindow):
	# Want to put the remove class and remove subjects windows into this window later on
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 300, 200
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Rename Class") # Window Title
		self.window = editer()
		self.setCentralWidget(self.window)

class editer(Wig.QWidget):
	def __init__(self):
		super().__init__()
		self.back_path = path.split('/')
		self.back_path = self.back_path[:-1]
		self.back_path = '/'.join(self.back_path)

		grid = Wig.QGridLayout(self)

		self.edit_select = Wig.QComboBox(self)
		self.class_map = pd.read_csv(self.back_path + '/class_map.csv', header=0, index_col=0)
		classes = self.class_map['Class'].values
		for i in classes:
			self.edit_select.addItem(i)

		self.rename      = Wig.QLineEdit(self)
		self.button      = Wig.QPushButton('Rename')
		self.button.clicked.connect(self.rename_class)

		grid.addWidget(self.edit_select,0,0)
		grid.addWidget(self.rename, 1,0)
		grid.addWidget(self.button,2,0)

	def rename_class(self):
		new_name = self.rename.text()
		self.class_map[self.class_map['Class'] == self.edit_select.currentText()] = new_name

		self.data_map = pd.read_csv(self.back_path + '/data_map.csv', header=0, index_col=0)
		self.data_map['Class'][self.data_map['Class'] ==self.edit_select.currentText()] = new_name

		self.class_map.to_csv(self.back_path + '/class_map.csv', header=True, index=True)
		self.data_map.to_csv(self.back_path + '/data_map.csv', header=True, index=True)

		os.rename(path + '/{}'.format(self.edit_select.currentText()), path + '/{}'.format(new_name))

#------------------------------------------------------------
class EDIT_SUBJECT(Wig.QMainWindow):
	# Want to put the remove class and remove subjects windows into this window later on
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 300, 200
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Rename Class") # Window Title
		self.window = editer2()
		self.setCentralWidget(self.window)

class editer2(Wig.QWidget):
	def __init__(self):
		super().__init__()
		self.back_path = path.split('/')
		self.back_path = self.back_path[:-1]
		self.back_path = '/'.join(self.back_path)

		grid = Wig.QGridLayout(self)

		self.edit_select = Wig.QComboBox(self)
		self.class_map = pd.read_csv(self.back_path + '/class_map.csv', header=0, index_col=0)
		classes = self.class_map['Class'].values
		for i in classes:
			self.edit_select.addItem(i)
		self.edit_select.currentTextChanged.connect(self.subject_shift)		

		self.subject_select = Wig.QComboBox(self)
		self.data_map = pd.read_csv(self.back_path + '/data_map.csv', header=0, index_col=0)
		self.subs = self.data_map['Subject'][self.data_map['Class'] == self.edit_select.currentText()]
		for i in self.subs:
			self.subject_select.addItem(i)
		

		self.rename      = Wig.QLineEdit(self)
		self.button      = Wig.QPushButton('Rename')
		self.button.clicked.connect(self.rename_subject)

		grid.addWidget(self.edit_select,0,0)
		grid.addWidget(self.subject_select,1,0)
		grid.addWidget(self.rename, 2,0)
		grid.addWidget(self.button,3,0)

	def subject_shift(self):
		self.subject_select.clear()
		self.subs = self.data_map['Subject'][self.data_map['Class'] == self.edit_select.currentText()]
		for i in self.subs:
			self.subject_select.addItem(i)


	def rename_subject(self):
		new_name = self.rename.text()
		mask = (self.data_map['Class'] == self.edit_select.currentText()) & (self.data_map['Subject'] == self.subject_select.currentText())
		self.data_map['Subject'][mask] = new_name

		self.data_map.to_csv(self.back_path + '/data_map.csv', header=True, index=True)

		os.rename(path + '/{}/{}'.format(self.edit_select.currentText(), self.subject_select.currentText()), path + '/{}/{}'.format(self.edit_select.currentText(), new_name))
		# Change this General Data as well.
		self.general_data = pd.read_csv(path + '/{}/{}/{}'.format(self.edit_select.currentText(), new_name, 'general_data.csv'), header=0, index_col=0)
		self.general_data['Subjects'][self.general_data['Subjects'] == self.subject_select.currentText()] = new_name
		self.general_data.to_csv(path + '/{}/{}/{}'.format(self.edit_select.currentText(), new_name, 'general_data.csv'), header=True, index=True)


#------------------------------------------------------------
class MOVE_CARDS(Wig.QMainWindow):
	# Want to put the remove class and remove subjects windows into this window later on
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 500, 800
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Rename Class") # Window Title
		self.window = editer3()
		self.setCentralWidget(self.window)

class editer3(Wig.QWidget):
	def __init__(self):
		super().__init__()
		self.back_path = path.split('/')
		self.back_path = self.back_path[:-1]
		self.back_path = '/'.join(self.back_path)

		grid = Wig.QGridLayout()
		vlayout = Wig.QVBoxLayout(self)

		self.class_map = pd.read_csv(self.back_path + '/class_map.csv', header=0, index_col=0)
		self.data_map  = pd.read_csv(self.back_path + '/data_map.csv', header=0, index_col=0)
		classes = self.class_map['Class'].values

		self.from_class = Wig.QComboBox(self)
		for i in classes:
			self.from_class.addItem(i)

		self.from_class.currentTextChanged.connect(self.subject_shift1)	

		self.to_class = Wig.QComboBox(self)
		for i in classes:
			self.to_class.addItem(i)

		self.to_class.currentTextChanged.connect(self.subject_shift2)	

		self.from_subject = Wig.QComboBox(self)
		
		self.subs1 = self.data_map['Subject'][self.data_map['Class'] == self.from_class.currentText()]
		for i in self.subs1:
			self.from_subject.addItem(i)

		self.from_subject.currentTextChanged.connect(self.table_shift1)	

		self.to_subject = Wig.QComboBox(self)
		self.subs2 = self.data_map['Subject'][self.data_map['Class'] == self.to_class.currentText()]
		for i in self.subs2:
			self.to_subject.addItem(i)

		self.to_subject.currentTextChanged.connect(self.table_shift2)	

		self.data1 = pd.read_csv(path + '/{}/{}/flashcards.csv'.format(self.from_class.currentText(), self.from_subject.currentText()), header=0, index_col=0)
		self.data2 = pd.read_csv(path + '/{}/{}/flashcards.csv'.format(self.to_class.currentText(), self.to_subject.currentText()), header=0, index_col=0)
		self.model1 = TableModel(self.data1)
		self.model2 = TableModel(self.data2)	

		self.table1 = Wig.QTableView()
		self.table1.setModel(self.model1)
		self.table2 = Wig.QTableView()
		self.table2.setModel(self.model2)	

		button = Wig.QPushButton('Move Cards Over')
		button.clicked.connect(self.move_cards_over)

		grid.addWidget(self.from_class, 0,0)
		grid.addWidget(self.from_subject, 1,0)
		grid.addWidget(self.to_class, 0,1)
		grid.addWidget(self.to_subject,1,1)	
		grid.addWidget(self.table1, 2,0)
		grid.addWidget(self.table2, 2,1)

		vlayout.addLayout(grid)
		vlayout.addWidget(button)

	def subject_shift1(self):
		self.from_subject.clear()
		self.subs1 = self.data_map['Subject'][self.data_map['Class'] == self.from_class.currentText()]
		for i in self.subs1:
			self.from_subject.addItem(i)
			

	def subject_shift2(self):
		self.to_subject.clear()
		self.subs2 = self.data_map['Subject'][self.data_map['Class'] == self.to_class.currentText()]
		for i in self.subs2:
			self.to_subject.addItem(i)


	def table_shift1(self):
		try:
			self.data1 = pd.read_csv(path + '/{}/{}/flashcards.csv'.format(self.from_class.currentText(), self.from_subject.currentText()), header=0, index_col=0)
			self.model1 = TableModel(self.data1)
			self.table1.setModel(self.model1)
			self.table1.repaint()
		except:
			pass	

	def table_shift2(self):
		try:
			self.data2 = pd.read_csv(path + '/{}/{}/flashcards.csv'.format(self.to_class.currentText(), self.to_subject.currentText()), header=0, index_col=0)
			self.model2 = TableModel(self.data2)
			self.table2.setModel(self.model2)
			self.table2.repaint()
		except:
			pass	

	def move_cards_over(self):
		try:
			indexes = self.table1.selectionModel().selectedRows()
			all_indexes = []
			for i in indexes:
				all_indexes += [i.row()]


			self.transplant = self.data1.iloc[all_indexes, :]
			self.data2 = self.data2.append(self.transplant)
			self.data2 = self.data2.reset_index(drop=True)
			self.data1 = self.data1.drop(self.data1.index[all_indexes])
		
			qs   = self.transplant['Term'].values
			ases = self.transplant['Definition'].values

			for i in qs:
				check = i.split('.')
				if check[-1] == 'png' or check[-1] == 'mov' or check[-1] == 'mp4' or check[-1] == 'MP3' or check[-1] == 'mp3':
					from_path = path + '/{}/{}/images/{}'.format(self.from_class.currentText(), self.from_subject.currentText(), i)
					to_path   = path + '/{}/{}/images/{}'.format(self.to_class.currentText(), self.to_subject.currentText(), i)

					shutil.move(from_path, to_path)
				else:
					pass

			for i in ases:
				check = i.split('.')
				if check[-1] == 'png' or check[-1] == 'mov' or check[-1] == 'mp4' or check[-1] == 'MP3' or check[-1] == 'mp3':
					from_path = path + '/{}/{}/images/{}'.format(self.from_class.currentText(), self.from_subject.currentText(), i)
					to_path   = path + '/{}/{}/images/{}'.format(self.to_class.currentText(), self.to_subject.currentText(), i)

					shutil.move(from_path, to_path)
				else:
					pass

			self.data1.to_csv(path + '/{}/{}/flashcards.csv'.format(self.from_class.currentText(), self.from_subject.currentText()), header=True, index=True)
			self.data2.to_csv(path + '/{}/{}/flashcards.csv'.format(self.to_class.currentText(), self.to_subject.currentText()), header=True, index=True)

			self.table_shift1()
			self.table_shift2()


		except:
			pass	


#------------------------------------------------------------
class REMOVE_CLASS(Wig.QMainWindow):
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 300, 100
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Remove Class") # Window Title
		self.window = remover()
		self.setCentralWidget(self.window)


class remover(Wig.QWidget):
	def __init__(self):
		super().__init__()
		self.back_path = path.split('/')
		self.back_path = self.back_path[:-1]
		self.back_path = '/'.join(self.back_path)

		try:
			self.class_map = pd.read_csv(self.back_path + '/class_map.csv', header=0, index_col=0)
			self.data_map  = pd.read_csv(self.back_path + '/data_map.csv', header=0, index_col=0)
			classes = self.class_map['Class'].values
			grid = Wig.QVBoxLayout(self)

			label = Wig.QLabel('Select Class to Remove')
			label.setFixedHeight(10)
			label.setAlignment(Qt.Qt.AlignCenter)

			self.input_line = Wig.QComboBox(self)
			for i in classes:
				self.input_line.addItem(i)

			enter = Wig.QPushButton('Enter')
			enter.clicked.connect(self.remove_class_now)

			grid.addWidget(label)
			grid.addWidget(self.input_line)
			grid.addWidget(enter)
			grid.addStretch(0)

		except:
			self.w = alert_win('Error: No classes to delete')
			self.show()

	def remove_class_now(self):
		
		to_remove = path + '/{}'.format(self.input_line.currentText())
		shutil.rmtree(to_remove)

		mask = (self.class_map['Class'] != self.input_line.currentText())
		self.class_map = self.class_map[mask]
		
		self.class_map.to_csv(self.back_path + '/class_map.csv', header=True, index=True)
		mask = (self.data_map['Class'] != self.input_line.currentText())
		self.data_map = self.data_map[mask]
		self.data_map.to_csv(self.back_path+'/data_map.csv', header=True, index=True)



#------------------------------------------------------------
class REMOVE_SUBJECT(Wig.QMainWindow):
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 300, 100
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Remove Subject") # Window Title
		self.window = remover2()
		self.setCentralWidget(self.window)


class remover2(Wig.QWidget):
	def __init__(self):
		super().__init__()
		grid = Wig.QVBoxLayout(self)

		self.back_path = path.split('/')
		self.back_path = self.back_path[:-1]
		self.back_path = '/'.join(self.back_path)

		try:
			self.data_map  = pd.read_csv(self.back_path + '/data_map.csv', header=0, index_col=0)
			self.class_map = pd.read_csv(self.back_path + '/class_map.csv', header=0, index_col=0)

			classes = self.class_map['Class'].values

			label = Wig.QLabel('Select Subject to Remove')
			label.setFixedHeight(10)
			label.setAlignment(Qt.Qt.AlignCenter)

			self.input_line = Wig.QComboBox(self)
			for i in classes:
				self.input_line.addItem(i)
			self.input_line.currentTextChanged.connect(self.shift)	


			self.subject_select = Wig.QComboBox(self)
		

			self.new_data_map = self.data_map[self.data_map['Class'] == self.input_line.currentText()]
			subjects = self.new_data_map['Subject'].values
			for i in subjects:
				self.subject_select.addItem(i)


			enter = Wig.QPushButton('Enter')
			enter.clicked.connect(self.remove_subject_now)

			grid.addWidget(label)
			grid.addWidget(self.input_line)
			grid.addWidget(self.subject_select)
			grid.addWidget(enter)
			grid.addStretch(0)

		except:
			self.w = alert_win('Error: Either no classes or no subjects to delete')
			self.show()

		

	def shift(self):
		self.subject_select.clear()
		self.new_data_map = self.data_map[self.data_map['Class'] == self.input_line.currentText()]
		subjects = self.new_data_map['Subject'].values
		for i in subjects:
			self.subject_select.addItem(i)


	def remove_subject_now(self):
		
		to_remove = path + '/{}/{}'.format(self.input_line.currentText(), self.subject_select.currentText())
		shutil.rmtree(to_remove)
		# This needs to be fixed
		mask = (self.data_map['Class'] != self.input_line.currentText()) | (self.data_map['Subject'] != self.subject_select.currentText())
		self.data_map = self.data_map[mask]
		
		self.data_map.to_csv(self.back_path + '/data_map.csv', header=True, index=True)		
		


#------------------------------------------------------------
class ADD_FOCUS(Wig.QMainWindow):
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 800, 400
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Add Focus") # Window Title
		self.window = adder3()
		self.setCentralWidget(self.window)


class adder3(Wig.QWidget):
	def __init__(self):
		super().__init__()
		back_path = path.split('/')
		back_path = back_path[:-1]
		self.back_path = '/'.join(back_path)
		#print(back_path)
		try:
			self.data = pd.read_csv(self.back_path+'/focus.csv', header=0, index_col=0)
		except:
			self.data = pd.DataFrame({'Class':[], 'Subject':[], 'Color':[]})
			self.data.to_csv(self.back_path+'/focus.csv',index=True)	

		grid = Wig.QVBoxLayout()
		grid2 = Wig.QVBoxLayout()
		hgrid = Wig.QHBoxLayout(self)

		self.class_wheel = Wig.QComboBox(self)
		classes = list(os.walk(os.getcwd()))[0][1]
		classes = np.sort(classes)
		for i in classes:
			self.class_wheel.addItem(i)
		self.class_wheel.currentTextChanged.connect(self.change_subject_wheel)

		self.colors = Wig.QComboBox(self)
		cols = ['Green', 'Blue', 'Yellow', 'Dark Green', 'Black']
		for i in cols:
			self.colors.addItem(i)
		self.colors.currentTextChanged.connect(self.change_focus_list)	

		self.subject_list = Wig.QListWidget(self)
		try:
			self.change_subject_wheel()
		except:
			pass

		add_button = Wig.QPushButton('ADD')
		add_button.clicked.connect(self.adding_focus)
		remove_button = Wig.QPushButton('REMOVE')
		remove_button.clicked.connect(self.remove_focus)

		self.focused_list = Wig.QListWidget(self)
		try:
			self.change_focus_list()
		except:
			pass			

		grid.addWidget(self.class_wheel)
		grid.addWidget(self.subject_list)
		grid.addWidget(add_button)
		grid.addWidget(remove_button)

		grid2.addWidget(self.colors)
		grid2.addWidget(self.focused_list)

		hgrid.addLayout(grid)
		hgrid.addLayout(grid2)

	def change_subject_wheel(self):
		self.subject_list.clear()
		self.class_wheel.repaint()
		call = file_walker(path).subjects(self.class_wheel.currentText())
		data_scoop = data_reader(self.class_wheel.currentText(), call)

		# This wont work on old material names when comnbining files!
		Ls, Fs = data_scoop.determine_L_F()
		
		for i in Ls:
			self.subject_list.addItem(i)
		
		for i in Fs:
			self.subject_list.addItem(i)
		self.subject_list.sortItems()		
		self.subject_list.repaint()

	def adding_focus(self):
		try:
			data_splice = pd.DataFrame({'Class':[self.class_wheel.currentText()] ,'Subject':[self.subject_list.currentItem().text()] , 'Color':[self.colors.currentText()]})
			self.data = self.data.append(data_splice, ignore_index=True, sort=False)
			self.data.to_csv(self.back_path+'/focus.csv', index=True)

			self.focused_list.addItem('{}--{}'.format(self.class_wheel.currentText(), self.subject_list.currentItem().text()))
		except:
			pass

	def remove_focus(self):
		indexer = self.focused_list.currentItem().text()
		indexer = indexer.split('--')
		mask = (self.data['Class'] != self.class_wheel.currentText()) & (self.data['Subject'] != indexer[1])
		self.data = self.data[mask]
		self.data.to_csv(self.back_path+'/focus.csv', index=True)

		self.focused_list.takeItem(self.focused_list.currentRow())

	def change_focus_list(self):
		self.focused_list.clear()
		self.data = pd.read_csv(self.back_path+'/focus.csv', header=0, index_col=0)
		for i in self.data.values:
			if i[-1] == self.colors.currentText():
				self.focused_list.addItem('{}--{}'.format(i[0], i[1]))
			else:
				pass	


#=========================================
#	       Machine Learning
#=========================================		
path = os.getcwd()

class machine_learning_win(Wig.QMainWindow):
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 500, 500
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Add Class") # Window Title
		self.window = mach1()
		self.setCentralWidget(self.window)


class mach1(Wig.QWidget):
	def __init__(self):
		super().__init__()

		back_path = path.split('/')[:-1]
		back_path = '/'.join(back_path)

		grid = Wig.QGridLayout(self)

		self.label_a = Wig.QLabel('a MSE: ')
		self.label_a.setAlignment(Qt.Qt.AlignCenter)

		self.table = Wig.QTableView()

		data = pd.read_csv(back_path + '/mach_data_map.csv', header=0, index_col=0)
		self.model = TableModel_norm(data)
		self.table.setModel(self.model)

		self.sc    = Canvas(self, width=5, height=3)

		self.button     = Wig.QPushButton('Initiate Learning')
		self.button.clicked.connect(self.run_algorithm)

		grid.addWidget(self.table,0,0)
		grid.addWidget(self.label_a, 1,0)
		grid.addWidget(self.sc,3,0)
		grid.addWidget(self.button,4,0)

	def run_algorithm(self):
		# This one is outdated!
		results = machine_learning_algorithm().Multioutput_Regressor()
		self.label_a.setText('a MAE: {}'.format(results[1]))

		self.sc.axes.clear()
		self.sc.axes.scatter(results[2], results[3][:,0], s=6, label='s-test')
		self.sc.axes.plot(results[2], results[4][:,0], label="s-pred")
		self.sc.axes.legend()
		self.sc.draw()
		self.sc.repaint()


class machine_learning_algorithm:
	# Class for Machine Learning for now or future machine learning
	# applications
	def __init__(self):
		self.call = file_walker(path)
		self.all_classes = self.call.classes
		self.all_subjects = [self.call.subjects(i) for i in self.all_classes]

	def Multioutput_Regressor(self):
		back_path = path.split('/')[:-1]
		back_path = '/'.join(back_path)

		mach_data = pd.read_csv(back_path + '/mach_data_map.csv', index_col=0, header=0)
		mach_data = mach_data.sort_values(by='class')


		decrypt = pd.read_csv(back_path+ '/crypts.csv', index_col=0, header=0)

		to_shift = mach_data['class'].values.flatten()
		
		new_vals = []
		for i in to_shift:
			new = decrypt[decrypt['uniques']==i]
			new_vals += [new['labels'].values[0]]
		
		
		mach_data['class'] = new_vals

		# Will need to change this data down here when I get enough Data that is

		in_Frame = mach_data.iloc[:,0:9]
		out_Frame = mach_data.iloc[:,9:]

		X = in_Frame.values
		Y = out_Frame.values

		xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size = 0.15)

		in_dim = X.shape[1]
		out_dim = Y.shape[1]

		model = Sequential()
		model.add(Dense(1000, input_dim=in_dim, activation="relu"))
		model.add(Dense(100, activation="relu"))
		model.add(Dropout(0.2))
		model.add(Dense(50, activation='swish'))
		model.add(Dense(10, activation='linear'))
		model.compile(loss="mse", optimizer="adam")

		model.fit(xtrain, ytrain, epochs=100, batch_size=20, verbose=0)
		model.save(back_path + '/machine_model.h5')

		ypred = model.predict(xtest)
		MAE_a = mean_absolute_error(ytest[:,0], ypred[:,0])

		x_ax = range(len(xtest))

		return model, MAE_a, x_ax, ytest, ypred   

#=========================================
#	          Default Time
#=========================================	

class Default_Time(Wig.QMainWindow):
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 800, 500
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Adjust Default Time Lapse") # Window Title
		self.window = time_lapse()
		self.setCentralWidget(self.window)

class time_lapse(Wig.QWidget):
	def __init__(self):
		super().__init__()
		grid = Wig.QGridLayout(self)

		self.sc = Canvas(self, height=3, width=3)
		self.label = Wig.QLabel()

		self.slider = Wig.QSlider(Qt.Qt.Horizontal)
		self.slider.setMinimum(1)
		self.slider.setMaximum(40)
		self.slider.setValue(10)
		self.slider.setTickPosition(Wig.QSlider.TicksBelow)
		self.slider.setTickInterval(5)

		self.slider2 = Wig.QSlider(Qt.Qt.Horizontal)
		self.slider2.setMinimum(1)
		self.slider2.setMaximum(20)
		self.slider2.setValue(2)
		self.slider2.setTickPosition(Wig.QSlider.TicksBelow)
		self.slider2.setTickInterval(5)

		self.slider.valueChanged.connect(self.plotter)
		self.slider2.valueChanged.connect(self.plotter)

		select = Wig.QPushButton('Select')
		select.clicked.connect(self.change_call)

		grid.addWidget(self.sc, 0, 0)
		grid.addWidget(self.label, 1, 0)
		grid.addWidget(self.slider, 2, 0)
		grid.addWidget(self.slider2, 3, 0)
		grid.addWidget(select, 4, 0)

	def plotter(self):
		self.sc.axes.clear()
		maintain = access_settings(R'settings.csv', change = 0)
		maintain = maintain['maintain'].values[0]
		maintain = maintain/100

		A = self.slider.value()
		B = self.slider2.value()

		s = []
		for i in range(5):
			s += [update_function_adjust(i, A, B)]

		curve_builder = []
		for i in s:
			y = norm_forgetting_curve(1, 0, i)
			y = y[y > maintain]
			curve_builder += [y]
		curve = list(np.concatenate(curve_builder).flat)

		peaks = find_peaks(curve)
		days = peaks[0]/10
		
		self.label.setText('Days till you hit lower percentage after each testing: {}'.format(days))
		
		x = time_builder(curve)

		self.sc.axes.plot(x, curve)	
		self.sc.draw()

	def change_call(self):
		path = os.getcwd()
		back_path = path.split('/')[:-1]
		back_path = '/'.join(back_path)

		A = self.slider.value()
		B = self.slider2.value()
		

		put_into = pd.DataFrame({'A':[A], 'B':[B]})
		put_into.to_csv(back_path + '/time_lapse.csv', header=True, index=True)

		try:
			# moves away from machine learning
			pull_out = back_path + '/machine_model.h5'
			os.remove(pull_out)	
		except:
			pass	


class IMPORT_data(Wig.QMainWindow):
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 600, 500
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Import Data") # Window Title
		self.window = importer()
		self.setCentralWidget(self.window)

class importer(Wig.QWidget):
	def __init__(self):
		super().__init__()
		self.path = os.getcwd()
		self.back_path = self.path.split('/')[:-1]
		self.back_path = '/'.join(self.back_path)

		try:
			self.class_map = pd.read_csv(self.back_path + '/class_map.csv', header=0, index_col=0)
			self.data_map  = pd.read_csv(self.back_path + '/data_map.csv', header=0, index_col=0)
		except:
			self.class_map = pd.DataFrame({'Class':[]})
			self.data_map  = pd.DataFrame({'Class':[], 'Subject':[], 'L_F':[]})

		grid = Wig.QGridLayout(self)

		self.select = Wig.QComboBox(self)
		self.select.addItem('Class')
		self.select.addItem('Subject')
		self.select.addItem('From Back Up')

		self.select.currentTextChanged.connect(self.switch)

		self.in_case_class = Wig.QComboBox(self)

		enter = Wig.QPushButton('Select File')
		enter.clicked.connect(self.import_now)

		grid.addWidget(self.select, 0, 0)
		grid.addWidget(self.in_case_class, 1, 0)
		grid.addWidget(enter, 2,0)

	def switch(self):
		try:
			if self.select.currentText() == 'Class' or self.select.currentText() == 'From Back Up':
				self.in_case_class.clear()

			else:
				self.in_case_class.clear()
				to_put = self.class_map['Class'].values
				for i in to_put:
					self.in_case_class.addItem(i)
		except:
			pass

	def import_now(self):
		
		here = self.path.split('/')
		here = here[0]

		pull = Wig.QFileDialog.getExistingDirectory(self, 'Select Import File', here)
		name = pull.split('/')
		# Name is the name of the class if class is selected
		# Name is the name of the subject if subject is selected
		name = name[-1]

		if self.select.currentText() == 'Class':
			try:
				check = file_walker(pull).classes
				shutil.move(pull, self.path + '/{}'.format(name))
				new = pd.DataFrame({'Class':[name]})
				self.class_map = self.class_map.append(new, ignore_index=True)
				self.class_map.to_csv(self.back_path + '/class_map.csv', header=True, index=True)

			# Need to add all files to the data_map
			
				class_name_list = [name] * len(check)
				l_list = ['L'] * len(check)

				new = pd.DataFrame({'Class':class_name_list, 'Subject':check, 'L_F':l_list})
				self.data_map = self.data_map.append(new, ignore_index=True)
				self.data_map.to_csv(self.back_path + '/data_map.csv', header=True, index=True)
			except:
				self.w = alert_win('Error occured: File Name already exists, try renaming the file')
				self.w.show()

		elif self.select.currentText() == 'From Back Up':
			# Need to select Back Up File This is a Factory Reset
			# Requires Reboot
			try:
				data_path  = pull + '/Data'
				class_path = pull + '/class_map.csv'
				crypt_path = pull + '/crypts.csv'
				datas_path = pull + '/data_map.csv'
				focus_path = pull + '/focus.csv'
				mach_path  = pull + '/mach_data_map.csv'
				setti_path = pull + '/settings.csv'
				time_path  = pull + '/time_lapse.csv'

				data_new   = self.back_path + '/Data'
				class_new  = self.back_path + '/class_map.csv'
				crypt_new  = self.back_path + '/crypts.csv'
				datas_new  = self.back_path + '/data_map.csv'
				focus_new  = self.back_path + '/focus.csv'
				mach_new   = self.back_path + '/mach_data_map.csv'
				setti_new  = self.back_path + '/settings.csv'
				time_new   = self.back_path + '/time_lapse.csv'

				if os.path.exists(data_new):
					shutil.rmtree(data_new)
			
				shutil.copytree(data_path, data_new)
				shutil.copy(class_path, class_new)
				shutil.copy(crypt_path, crypt_new)
				shutil.copy(datas_path, datas_new)
				shutil.copy(focus_path, focus_new)
				shutil.copy(mach_path, mach_new)
				shutil.copy(setti_path, setti_new)
				shutil.copy(time_path, time_new)
			except:
				pass	


		else:
			try:
				shutil.move(pull, self.path + '/{}/{}'.format(self.in_case_class.currentText(), name))

				new = pd.DataFrame({'Class':[self.in_case_class.currentText()], 'Subject':[name], 'L_F':['L']})
				self.data_map = self.data_map.append(new, ignore_index=True)
				self.data_map.to_csv(self.back_path + '/data_map.csv', header=True, index=True)
			except:
				self.w = alert_win('Error occured: File Name already exists, try renaming the file')
				self.w.show()						


					


#=========================================
#	            Progress
#=========================================	
path = os.getcwd()

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class TRACKING_progress(Wig.QMainWindow):
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 700, 800
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Progress") # Window Title
		self.window = track1()
		self.setCentralWidget(self.window)


class track1(Wig.QWidget):
	def __init__(self):
		super().__init__()

		back_path = os.getcwd().split('/')[:-1]
		self.back_path = '/'.join(back_path)

		grid = Wig.QVBoxLayout(self)

		label1 = Wig.QLabel('Select Class')
		label1.setFixedHeight(10)
		label1.setAlignment(Qt.Qt.AlignCenter)	
		self.class_wheel = Wig.QComboBox(self)
		try:
			classes = pd.read_csv(self.back_path + '/class_map.csv', header=0, index_col=0)
			classes = classes['Class'].values
			for i in classes:
				self.class_wheel.addItem(i)	
		except:
			pass

		
		self.class_wheel.currentTextChanged.connect(self.subject_shift)
		
		self.sc = Canvas(self, width=3, height=3)
		self.table = Wig.QTableView()
		

		grid.addWidget(label1)
		grid.addWidget(self.class_wheel)
		grid.addWidget(self.sc)
		grid.addWidget(self.table)


	def subject_shift(self):
		
		self.sc.axes.clear()

		data_map = pd.read_csv(self.back_path + '/data_map.csv', header=0, index_col=0)
		all_subs = data_map[data_map['Class'] == self.class_wheel.currentText()]
		all_subs = all_subs['Subject'].values
		

		call = data_reader(self.class_wheel.currentText(), all_subs)
		progress_data = call.progress2()
		#print(progress_data)
		test_scores = progress_data['Test Score'].values.flatten()
		ave_score = np.mean(test_scores)
		ave_score = float('{:.2f}'.format(ave_score))
		estimated_scores = progress_data['Estimated Score'].values.flatten()
		ave_est = np.mean(estimated_scores)
		ave_est = float('{:.2f}'.format(ave_est))

		labels = ['Total Test Score', 'Total Estimated Score']
		data = [ave_score, ave_est]
	#number of data points
		n = len(data)
		percent_circle = max(data)

	#radius of donut chart
		r = 1.5
		r_inner = 0.4
	#calculate width of each ring
		w = (r-r_inner)/n

	#create colors along a chosen colormap
		colors = plt.cm.tab10.colors


		self.sc.axes.axis("equal")

	#create rings of donut chart
		for i in range(n):
			radius = r - i * w
		#hide labels in segments with textprops: alpha = 0 - transparent, alpha = 1 - visible
			try:
				self.sc.axes.pie([data[i] / max(data) * percent_circle], radius=radius, startangle=90,
								counterclock=False,
								colors=[colors[i]],
								labels=[f'{labels[i]} – {data[i]}'], labeldistance=None,
								wedgeprops={'width': w, 'edgecolor': 'white'},
								normalize=False)
			#self.sc.axes.pie([1], radius = w)
				self.sc.axes.text(0, radius - w / 2, f'{labels[i]} – {data[i]} ', ha='right', va='center')
			except:
				# This for when there is no scoreing involved or it is zero
				self.sc.axes.pie([data[i] * percent_circle], radius=radius, startangle=90,
								counterclock=False,
								colors=[colors[i]],
								labels=[f'{labels[i]} – {data[i]}'], labeldistance=None,
								wedgeprops={'width': w, 'edgecolor': 'white'})
			#self.sc.axes.pie([1], radius = w)
				self.sc.axes.text(0, radius - w / 2, f'{labels[i]} – {data[i]} ', ha='right', va='center')	

		self.sc.draw()
		self.repaint()

		self.model = TableModel(progress_data)
		self.table.setModel(self.model)
		self.table.repaint()

#-------------------------------------------------------------------------
#							Progress
#-------------------------------------------------------------------------
class TRACKING_ALL(Wig.QMainWindow):
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 1500, 1000
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Add Class") # Window Title
		self.window = track2()
		self.setCentralWidget(self.window)


class track2(Wig.QWidget):
	def __init__(self):
		super().__init__()

		grid = Wig.QHBoxLayout(self)
		
		self.sc = Canvas(self, width=3, height=3)
		self.sc2 = Canvas(self,width = 3,height = 3)

		grid.addWidget(self.sc)
		grid.addWidget(self.sc2)

		labels = []
		test_scores = []
		est_scores = []
		walk_call = file_walker(path)
		all_classes = walk_call.classes
		for i in all_classes:
			subs = walk_call.subjects(i)
			call = data_reader(i, subs)
			progress_data = call.progress()
			test = progress_data['Test Score'].values.flatten()
			ave_score = np.mean(test)
			ave_score = float('{:.2f}'.format(ave_score))

			estimated_scores = progress_data['Estimated Score'].values.flatten()
			ave_est = np.mean(estimated_scores)
			ave_est = float('{:.2f}'.format(ave_est))
			labels += [i]
			test_scores += [ave_score]
			est_scores += [ave_est]

		
		
	#number of data points
		n = len(test_scores)
		percent_circle = max(test_scores)

	#radius of donut chart
		r = 1.5
		r_inner = 0.4
	#calculate width of each ring
		w = (r-r_inner)/n

	#create colors along a chosen colormap
		#colors = plt.cm.tab20c.colors
		

		self.sc.axes.axis("equal")

	#create rings of donut chart
		for i in range(n):
			radius = r - i * w
		#hide labels in segments with textprops: alpha = 0 - transparent, alpha = 1 - visible
			try:
				self.sc.axes.pie([test_scores[i] / max(test_scores) * percent_circle], radius=radius, startangle=90,
								counterclock=False,
								labels=[f'{labels[i]} – {test_scores[i]}'], labeldistance=None,
								wedgeprops={'width': w, 'edgecolor': 'white'})
			#self.sc.axes.pie([1], radius = w)
				self.sc.axes.text(0, radius - w / 2, f'{labels[i]} – {test_scores[i]} ', ha='right', va='center')
			except:
				self.sc.axes.pie([test_scores[i] * percent_circle], radius=radius, startangle=90,
								counterclock=False,
								labels=[f'{labels[i]} – {test_scores[i]}'], labeldistance=None,
								wedgeprops={'width': w, 'edgecolor': 'white'})
			#self.sc.axes.pie([1], radius = w)
				self.sc.axes.text(0, radius - w / 2, f'{labels[i]} – {test_scores[i]} ', ha='right', va='center')
					

		self.sc.draw()
		self.repaint()

		#number of data points
		n = len(est_scores)
		percent_circle = max(est_scores)

	#radius of donut chart
		r = 1.5
		r_inner = 0.4
	#calculate width of each ring
		w = (r-r_inner)/n

	#create colors along a chosen colormap
		#colors = plt.cm.tab10.colors
		self.sc2.axes.axis("equal")

	#create rings of donut chart
		for i in range(n):
			radius = r - i * w
		#hide labels in segments with textprops: alpha = 0 - transparent, alpha = 1 - visible
			try:
				self.sc2.axes.pie([est_scores[i] / max(est_scores) * percent_circle], radius=radius, startangle=90,
								counterclock=False,
								labels=[f'{labels[i]} – {est_scores[i]}'], labeldistance=None,
								wedgeprops={'width': w, 'edgecolor': 'white'})
			#self.sc.axes.pie([1], radius = w)
				self.sc2.axes.text(0, radius - w / 2, f'{labels[i]} – {est_scores[i]} ', ha='right', va='center')
			except:
				self.sc2.axes.pie([est_scores[i] * percent_circle], radius=radius, startangle=90,
								counterclock=False,
								labels=[f'{labels[i]} – {est_scores[i]}'], labeldistance=None,
								wedgeprops={'width': w, 'edgecolor': 'white'})
			#self.sc.axes.pie([1], radius = w)
				self.sc2.axes.text(0, radius - w / 2, f'{labels[i]} – {est_scores[i]} ', ha='right', va='center')
					

		self.sc2.draw()
		self.repaint()

#-----------------------------------------
#					    Special Tests
#-----------------------------------------
class GIANT_TESTS(Wig.QMainWindow):
	# Want to put the remove class and remove subjects windows into this window later on
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 900, 800
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.setWindowTitle("Rename Class") # Window Title
		self.window = giant_test_window()
		self.setCentralWidget(self.window)

class giant_test_window(Wig.QWidget):
	# Goal is to make a GIANT TEST for Tutoring
	def __init__(self):
		super().__init__()

		instructions = "This window will create and start a new test by taking cards randomly from the selected subjects. \nCards can also be manually added or removed to the test as well"

		self.back_path = path.split('/')
		self.back_path = self.back_path[:-1]
		self.back_path = '/'.join(self.back_path)

		grid = Wig.QGridLayout()
		vlayout = Wig.QVBoxLayout(self)

		label = Wig.QLabel(instructions)

		self.class_map = pd.read_csv(self.back_path + '/class_map.csv', header=0, index_col=0)
		self.data_map  = pd.read_csv(self.back_path + '/data_map.csv', header=0, index_col=0)
		classes = self.class_map['Class'].values

		self.from_class = Wig.QComboBox(self)
		for i in classes:
			self.from_class.addItem(i)

		self.from_class.currentTextChanged.connect(self.subject_shift1)	

		self.from_subject = Wig.QComboBox(self)
		
		self.subs1 = self.data_map['Subject'][self.data_map['Class'] == self.from_class.currentText()]
		for i in self.subs1:
			self.from_subject.addItem(i)

		self.from_subject.currentTextChanged.connect(self.table_shift1)		

		self.data1 = pd.read_csv(path + '/{}/{}/flashcards.csv'.format(self.from_class.currentText(), self.from_subject.currentText()), header=0, index_col=0)
		self.data2 = pd.DataFrame({'Class':[], 'Subject':[], 'Term':[], 'Definition':[], 'Correct':[], 'Incorrect':[]})
		
		self.model1 = TableModel(self.data1)

		self.table1 = Wig.QTableView()
		self.table1.setModel(self.model1)

		test_label = Wig.QLabel('Test Cards')
		self.table2 = Wig.QTableView()

		self.move_selected = Wig.QPushButton('Move Selected')
		self.move_selected.clicked.connect(self.move_selected_over)
		self.how_many_random = Wig.QSpinBox()
		self.random_over = Wig.QPushButton("Random Amount Select")
		self.random_over.clicked.connect(self.move_over_random)

		button = Wig.QPushButton('Make Test')
		button.clicked.connect(self.make_test)

		grid.addWidget(self.from_class, 0,0)
		grid.addWidget(self.from_subject, 1,0)
		grid.addWidget(test_label,1,2)
		grid.addWidget(self.table1, 2,0)
		grid.addWidget(self.table2, 2,2)
		
		grid.addWidget(self.how_many_random,0,1)
		grid.addWidget(self.random_over,1,1)
		grid.addWidget(self.move_selected, 2,1)

		vlayout.addWidget(label)
		vlayout.addLayout(grid)
		vlayout.addWidget(button)

	def subject_shift1(self):
		self.from_subject.clear()
		self.subs1 = self.data_map['Subject'][self.data_map['Class'] == self.from_class.currentText()]
		for i in self.subs1:
			self.from_subject.addItem(i)
			
	def table_shift1(self):
		try:
			self.data1 = pd.read_csv(path + '/{}/{}/flashcards.csv'.format(self.from_class.currentText(), self.from_subject.currentText()), header=0, index_col=0)
			self.model1 = TableModel(self.data1)
			self.table1.setModel(self.model1)
			self.table1.repaint()
		except:
			pass

	def move_over_random(self):
		try:
			value = self.how_many_random.value()
			frame = self.data1.sample(n=value)
			to_know = len(frame)
			from_c = self.from_class.currentText()
			from_s = self.from_subject.currentText()
			frame.insert(0,'Class',[from_c]*to_know)
			frame.insert(1,'Subject',[from_s]*to_know)
			
			self.data2 = self.data2.append(frame)
			self.data2 = self.data2.reset_index(drop=True)

			self.model2 = TableModel(self.data2)
			self.table2.setModel(self.model2)
			self.table2.repaint()
		except:
			pass	

	def move_selected_over(self):
		try:
		
			indexes = self.table1.selectionModel().selectedRows()
			all_indexes = []
			for i in indexes:
				all_indexes += [i.row()]

			self.transplant = self.data1.iloc[all_indexes, :]
			to_know = len(self.transplant)
			from_c = self.from_class.currentText()
			from_s = self.from_subject.currentText()
			self.transplant.insert(0,'Class',[from_c]*to_know)
			self.transplant.insert(1,'Subject',[from_s]*to_know)

			self.data2 = self.data2.append(self.transplant)
			self.data2 = self.data2.reset_index(drop=True)
			
			self.model2 = TableModel(self.data2)
			self.table2.setModel(self.model2)
			self.table2.repaint()

		except:
			pass	



	def make_test(self):
		self.data2 = self.data2.sample(frac=1)
		self.data2 = frame_to_listframe(self.data2)
		#print(self.data2)
		self.w = test_for_giant_test(self.data2)
		self.w.show()


# May Need all of of this for compiation but running on python is too 
# heavy for CPU to handle

#from tensorflow import *

# This portion required for infinite loops
# created by Sklearn apparently


# Setting Up
#------------------------------

# Creating Data File and Entering Data File
#------------------------------


# setting file created
# Date checked and daily tracker changed if greater than a day
settings = access_settings(R'settings.csv', change = 0)
goals = settings['Goals'].values[0]
date = settings['date'].values[0]
maintain = settings['maintain'].values[0]
check_path = settings['text_size'].values[0]
difference = datetime.datetime.now() - datetime.datetime.strptime(date, '%Y-%m-%d')
difference = difference.days

if difference >= 1:
	settings = access_settings(R'settings.csv', change = [goals, 0, datetime.datetime.today().strftime('%Y-%m-%d'),maintain, check_path])
else:
	pass	


class App(Wig.QMainWindow):
	def __init__(self):
		super().__init__()
		self.x, self.y, self.w, self.h = 0, 0, 1500, 1000
		self.setGeometry(self.x, self.y, self.w, self.h)
		self.window = MainWindow(self)
		self.setCentralWidget(self.window)
		self.setWindowTitle("Study Application") # Window Title
		self.show()

		self.initUI()

	def initUI(self):
		#---------------------------------------
		bar = self.menuBar()

		fileMenu = bar.addMenu('File')
		dailyMenu = bar.addMenu('Settings')
		progressMenu = bar.addMenu('Progress')
		machine_learning = bar.addMenu('Machine Learning')
		data = bar.addMenu('Data')
		special_test = bar.addMenu('Special Tests')

		add_class = Wig.QAction('Add Class', self)
		add_class.triggered.connect(self.ADD_class)

		add_file = Wig.QAction('Add Subject', self)
		add_file.triggered.connect(self.ADD_file)

		edit_class = Wig.QAction('Rename Class', self)
		edit_class.triggered.connect(self.EDIT_class)

		edit_subject = Wig.QAction('Rename Subject', self)
		edit_subject.triggered.connect(self.EDIT_subject)

		move_cards = Wig.QAction('Move Cards', self)
		move_cards.triggered.connect(self.MOVE_cards)

		add_focus = Wig.QAction('Add Focus', self)
		add_focus.triggered.connect(self.ADD_focus)

		remove_class = Wig.QAction('Remove Class', self)
		remove_class.triggered.connect(self.REMOVE_class)

		remove_subject = Wig.QAction('Remove Subject', self)
		remove_subject.triggered.connect(self.REMOVE_subject)

		alter = Wig.QAction('Change Settings', self)
		alter.triggered.connect(self.change_settings)

		tracking_progress = Wig.QAction('Show Class Progress', self)
		tracking_progress.triggered.connect(self.TRACK_progress)

		default_time = Wig.QAction('Default Time Lapse', self)
		default_time.triggered.connect(self.DEFAULT_TIME)

		run_algorithm = Wig.QAction('Run Machine Learning Algorithm', self)
		run_algorithm.triggered.connect(self.machine_learning_window)

		data_back = Wig.QAction('Back Up Data', self)
		data_back.triggered.connect(self.DATA_BACKUP)

		data_share = Wig.QAction('Share Data', self)
		data_share.triggered.connect(self.DATA_SHARE)

		data_import = Wig.QAction('Import Data', self)
		data_import.triggered.connect(self.IMPORT_DATA)

		giant_test = Wig.QAction('Big Test', self)
		giant_test.triggered.connect(self.GIANT_TEST)

		fileMenu.addAction(add_class)
		fileMenu.addAction(add_file)
		fileMenu.addAction(edit_class)
		fileMenu.addAction(edit_subject)
		fileMenu.addAction(move_cards)
		fileMenu.addAction(add_focus)

		fileMenu.addAction(remove_class)
		fileMenu.addAction(remove_subject)

		dailyMenu.addAction(alter)
		dailyMenu.addAction(default_time)

		progressMenu.addAction(tracking_progress)

		data.addAction(data_back)
		data.addAction(data_share)
		data.addAction(data_import)

		machine_learning.addAction(run_algorithm)

		special_test.addAction(giant_test)
		#----------------------------------------

	def change_settings(self):
		self.w = changing_sett()
		self.w.show()

	def ADD_class(self):
		self.w = ADD_CLASS()
		self.w.show()

	def ADD_file(self):
		self.w = ADD_FILE()
		self.w.show()

	def EDIT_class(self):
		self.w = EDIT_CLASS()
		self.w.show()	

	def EDIT_subject(self):
		self.w = EDIT_SUBJECT()
		self.w.show()	

	def MOVE_cards(self):
		self.w = MOVE_CARDS()
		self.w.show()	

	def ADD_focus(self):
		self.w = ADD_FOCUS()
		self.w.show()

	def REMOVE_class(self):
		self.w = REMOVE_CLASS()
		self.w.show()

	def REMOVE_subject(self):	
		self.w = REMOVE_SUBJECT()
		self.w.show()		

	def arrange_files(self):
		self.w = arrange_nav_files()
		self.w.show()

	def combine_f(self):
		self.w = combine_files_func()
		self.w.show()

	def machine_learning_window(self):
		self.w = machine_learning_win()
		self.w.show()

	def TRACK_progress(self):
		self.w = TRACKING_progress()
		self.w.show()	

	def DEFAULT_TIME(self):
		self.w = Default_Time()
		self.w.show()

	def DATA_BACKUP(self):
		path = os.getcwd()
		back_path = path.split('/')[:-1]
		back_path = '/'.join(back_path)

		data      = back_path + '/Data'
		class_map = back_path + '/class_map.csv'
		crypts    = back_path + '/crypts.csv'
		data_map  = back_path + '/data_map.csv'
		focus     = back_path + '/focus.csv'
		mach_data = back_path + '/mach_data_map.csv'
		settings  = back_path + '/settings.csv'
		time_lapse= back_path + '/time_lapse.csv'

		here = path.split('/')
		here = here[0]

		pull = Wig.QFileDialog.getExistingDirectory(self, 'Select Placement Directory', here)
		pull = pull + '/BackUp_{}'.format(datetime.datetime.today().strftime("%d_%m_%Y"))
	
		shutil.copytree(data, pull + '/Data')
		shutil.copyfile(class_map, pull + '/class_map.csv')
		shutil.copyfile(crypts, pull + '/crypts.csv')
		shutil.copyfile(data_map, pull + '/data_map.csv')
		shutil.copyfile(focus, pull + '/focus.csv')
		shutil.copyfile(mach_data, pull + '/mach_data_map.csv')
		shutil.copyfile(settings, pull + '/settings.csv')
		shutil.copyfile(time_lapse, pull + '/time_lapse.csv')

	def DATA_SHARE(self):
		# Need to differentiate between class and subject
		# Need to redo files that contain curves....
		try:
			path = os.getcwd()

			here = path.split('/')
			here = here[0]
			pull = Wig.QFileDialog.getExistingDirectory(self, 'Select File to Share', path)
			file_name = pull.split('/')[-1]
			push = Wig.QFileDialog.getExistingDirectory(self, 'Select Placement Directory', here)
		
			shutil.copytree(pull, push + '/{}'.format(file_name))

		except:
			self.w = alert_win('Error in completion')
			self.w.show()	

	def IMPORT_DATA(self):
		# Doesnt work if it is the same way
		self.w = IMPORT_data()
		self.w.show()

	def GIANT_TEST(self):
		self.w = GIANT_TESTS()
		self.w.show()	


class MainWindow(Wig.QWidget):
	def __init__(self, parent):
		super(MainWindow, self).__init__(parent)
		layout = Wig.QVBoxLayout(self)
		layout.addWidget(Home(self))


if __name__ == '__main__':

	app = Wig.QApplication(sys.argv)
	ex = App()
	sys.exit(app.exec_())                