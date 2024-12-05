import datetime, pickle, cv2, register.util as util, os, sys
import reocognition.face_recognition_original as face_recognition_original
import numpy as np
import math

import sys
sys.path.insert(0, './Silent_Face_Anti_Spoofing')
from detection import test


def face_confidence(face_distance, face_match_threshold = 0.6):

	_range = (1.0 - face_match_threshold)
	linear_val = (1.0 - face_distance) / (_range * 2.0)

	if face_distance > face_match_threshold:
		return str(round(linear_val*100, 2)) + "%"
	
	else:
		value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5)*2, 0.2 ))) * 100

		return str(round(value, 2)) + "%"

class FaceDetection:
	db_dir = './data'
	db_paths = sorted(os.listdir(db_dir))
	log_path = './log.txt'

	face_locations = []
	face_encodings = []
	face_names = []
	known_face_encodings = []
	known_face_name = []

	faceDetect = cv2.CascadeClassifier(
		cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
	)

	def __init__(self):

		self.encode_faces()
		if not os.path.exists(self.db_dir):
			os.makedirs(self.db_dir)


		self.add_webcam()
	
	def encode_faces(self):
		self.known_face_encodings = []
		self.known_face_name = []

		for image in os.listdir(self.db_dir):
			face_image = face_recognition_original.load_image_file(f'{self.db_dir}/{image}')
			face_encoding = face_recognition_original.face_encodings(face_image)[0]

			self.known_face_encodings.append(face_encoding)
			self.known_face_name.append(image)
		
		print(self.known_face_name)

	def detect_face(self, frame):
		gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		face = self.faceDetect.detectMultiScale(
			gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
		)

		if len(face) > 0:
			return True
		else:
			return False

	
	def add_webcam(self):
		self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

		self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

		self.cam.set(cv2.CAP_PROP_FPS, 10)

		if not self.cam.isOpened():
			print("Video Source not found...")
			exit()

		count = 0
		count_detect = 0
		count_not_detect = 0

		while True:
			count += 1
			ret, frame = self.cam.read()

			if not ret:
				print("failed to grab frame")
				exit()

			if count % 30 == 0:
				if self.detect_face(frame):
					
					label, image_box = test(
						image = frame,
						model_dir = './Silent_Face_Anti_Spoofing/resources/anti_spoof_models',
						device_id = 0
					)

					print(label)
					
					if label != 1:
						count_detect = 0
						count_not_detect += 1
						if count_not_detect >= 5:
							try:
								os.remove("temp.png")
							except:
								pass
						print("spoofed")
						continue
					
					count_not_detect = 0
					count_detect += 1
					if count_detect >= 5:
						count_detect = 0
						cv2.imwrite("temp.png", frame)

				else:
					count_detect = 0
					count_not_detect += 1
					if count_not_detect >= 5:
						try:
							os.remove("temp.png")
						except:
							pass
					


			
			cv2.imshow('Face', frame)


			key = cv2.waitKey(1)

			if key == ord('q'):
				print('press')
				break
			
			if cv2.getWindowProperty('Face', cv2.WND_PROP_VISIBLE) <1:
				break
		os.remove("temp.png")
		self.cam.release()
		cv2.destroyAllWindows()
	
	
		
 

def main():
	app = FaceDetection()

if __name__ == "__main__":
	main()