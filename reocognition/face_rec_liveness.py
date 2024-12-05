import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cwd = os.getcwd()

import cv2
from deepface import DeepFace
import tkinter as tk
from PIL import Image, ImageTk

import tensorflow as tf
import numpy as np

# liveness model
model: tf.keras.Model = tf.keras.models.load_model('liveness.model')


def get_file(path: str):
	person, s = path.split('/')[-2:]
	s = s.split(".")

	if len(s) <= 0:
		raise ValueError
	
	if len(s) == 1:
		return s[0], None

	file_name_extension = s[-1]
	name = ".".join(s[:-1])

	return person, file_name_extension


def findThreshold(model_name, distance_metric):

    base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}

    thresholds = {
        "VGG-Face": {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86},
        "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
        "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
        "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
        "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
        "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
    }

    threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)

    return threshold

def face_condidence(distance, threshold = 0.6):
	range_ = (1.0 - threshold)
	linear_val = (1.0 - distance) / (range_ * 2.0)

	if distance > threshold:
		return f'{round(linear_val * 100, 2)}%'
	else:
		value = (linear_val + ((1.0 - linear_val) * pow((linear_val - 0.5)*2, 0.2))) * 100
		return f'{round(value, 2)}%'
	
class App:
	'''
		model_name = [
			"VGG-Face", 
			"Facenet", 
			"Facenet512", 
			"OpenFace", 
			"DeepFace", 
			"DeepID", 
			"ArcFace", 
			"Dlib", 
			"SFace",
		]

		distance_metric = ["cosine", "euclidean", "euclidean_l2"]

		detector_backend = [
			'opencv', 
			'ssd', 
			'dlib', 
			'mtcnn', 
			'retinaface', 
			'mediapipe',
			'yolov8',
			'yunet',
			'fastmtcnn',
		]
	'''

	# Path to store current frame
	TEMP_FILE = f'{cwd}/temp.jpg'

	# count frame
	count = 0

	def __init__(self, cam_ID: int, size, FPS: int, db_path: os.path, model_name = "Facenet512", distance_metric = 'cosine', detector_backend = 'opencv'):

		# cv2 init, get cam, set Width, Height, set FPS
		self.cam = cv2.VideoCapture(cam_ID)
		self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
		self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
		self.cam.set(cv2.CAP_PROP_POS_FRAMES, FPS)
		self.FPS = FPS

		# tkinter init, make screen
		self.main_screen = tk.Tk()
		self.main_screen.geometry(f"{size[0]+20}x{size[1]}")

		# canvas to show vid
		self.webcam_label = tk.Label(self.main_screen)
		self.webcam_label.grid(row=0, column=0)
		self.webcam_label.place(x = 10, y = 0, width = size[0], height=size[1])
		
		# face data path
		self.db_path = db_path

		# Deepace property
		self.model_name = model_name
		self.distance_metric = distance_metric
		self.detector_backend = detector_backend

		# if exist, remove
		file_name = f"representations_{self.model_name}.pkl"
		file_name = file_name.replace("-", "_").lower()
		try:
			os.remove(f'{self.db_path}/{file_name}')
			os.remove(self.TEMP_FILE)
		except:
			pass

		
		# initial create represent to pickle
		if len(os.listdir(self.db_path)) > 0:
			DeepFace.find(f'{cwd}/_temp.jpg', self.db_path, model_name=self.model_name, distance_metric=self.distance_metric, detector_backend=self.detector_backend, silent=True)
		else:
			print("no_data")

		self.add_webcam(self.webcam_label)
	
	def find_face(self):
		'''
		COMPARE face in TEMP_FILE if face in db_path
		raise ValueError if cannot find any face in TEMP_FILE
		'''
		
		# deepface lib
		detect = DeepFace.find(self.TEMP_FILE, self.db_path, model_name=self.model_name, distance_metric=self.distance_metric, enforce_detection=False, detector_backend=self.detector_backend, silent=True)
		data = []


		
		for info in detect:
			if len(info['identity']) > 1:
				name1 = get_file(info['identity'][0])[0]
				name2 = get_file(info['identity'][1])[0]
				if name1 == name2:
					name = name1
				else:
					name = "Unknown"

			else:
				name = 'Unknown'


			x1, y1 = info['source_x'][0], info['source_y'][0]
			x2, y2 = x1 + info['source_w'][0], y1 + info['source_h'][0]

			# liveness
			_img = cv2.imread(self.TEMP_FILE)[x1: x2, y1:y2]
			_img = cv2.resize(_img, (32, 32))
			_img = _img.astype('float')/255.0
			_img = tf.keras.preprocessing.image.img_to_array(_img)
			_img = np.expand_dims(_img, axis = 0)

			liveness = model.predict(_img, verbose=0)
			liveness = liveness[0].argmax()
			# end liveness

			distance = info[f'{self.model_name}_{self.distance_metric}'][0]
			threshold = findThreshold(self.model_name, self.distance_metric)
			confidence = face_condidence(distance, threshold)

			data.append( (name, (x1, y1), (x2, y2), confidence, liveness ))

		return data
	
	def recognition(self, frame, rec_per_frame):
		'''
		main recognition function
		'''
		# check every {rec_per_frame} frame
		if self.count % rec_per_frame == 0:
			self.count = 0

			# write frame to file
			cv2.imwrite(self.TEMP_FILE, frame)

			try:
				self.data = self.find_face()

			except ValueError:
				print("cant read face")

		# Draw face box
		for name, s, e, confidence, liveness in self.data:
			cv2.rectangle(frame, s, e, (0, 255, 0))
			cv2.rectangle(frame, (s[0], s[1]-40), (e[0], s[1]), (0, 255, 0), -1)

			if name == 'Unknown':
				cv2.putText(frame, f'{name}', (s[0], s[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

			else:
				cv2.putText(frame, f'{name}: {liveness}', (s[0], s[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
		
		
	def add_webcam(self, label: tk.Label):
		'''
		add cv2 to tkinter
		'''
		
		self.data = []

		self._label = label

		self.process_webcam()
	
	def process_webcam(self):
		self.count += 1

		# read from cam
		ret, frame = self.cam.read()

		# cannot grab frame
		if not ret:
			print("failed to grab frame")
			exit()

		# main recognition function
		self.recognition(frame, 10)
		
		
		#from frame to tkinter
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(img)
		imgtk = ImageTk.PhotoImage(image=img)
		self._label.imgtk = imgtk
		self._label.config(image=imgtk)

		# mainloop FPS
		self._label.after(self.FPS, self.process_webcam)
	
	def run(self):
		# tkinter mainloop
		self.main_screen.mainloop()
		# exit
		self.exit()
	

	def exit(self):
		'''
		release cam and remove temp file
		'''
		self.cam.release()

		file_name = f"representations_{self.model_name}.pkl"
		file_name = file_name.replace("-", "_").lower()
		try:
			os.remove(f'{self.db_path}/{file_name}')
			os.remove(self.TEMP_FILE)
		except:
			pass

def main():
	face_reg = App(
			cam_ID= 0,
			size= (640, 480),
			FPS= 30,
			db_path= f'{cwd}/data',
			model_name= "Facenet",
			distance_metric= "cosine",
			detector_backend= "opencv"
		)

	face_reg.run()


if __name__ == "__main__":
	main()