
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tkinter as tk
import cv2
import util as util
from PIL import Image, ImageTk
from deepface import DeepFace

import tensorflow as tf
import numpy as np

cwd = os.getcwd()


class Resigter:
	db_dir = f'{cwd}/data'
	def __init__(self, cam_id = 0):
		self.cam_id = cam_id

		if not os.path.exists(self.db_dir):
			os.makedirs(self.db_dir)

		self.main_screen = tk.Tk()
		self.main_screen.geometry("1460x520")
		self.register_button = util.get_button(self.main_screen, "Register", 'green', (550, 400), self.register, 'white')
		self.main_screen.bind('<Return>', self.temp)
		self.webcam_label = util.get_img_label(self.main_screen, (10, 0), (500, 500))

		self.add_webcam(self.webcam_label)
	
	def reset(self):
		self.register_button = util.get_button(self.main_screen, "Register", 'green', (550, 400), self.register, 'white')

		self.acpt_register_button.destroy()
		self.try_again_register_button.destroy()
		self.text_label_register.destroy()
		self.entry_text_register.destroy()
		self.capture_label.destroy()

	
	def add_webcam(self, label: tk.Label):
		self.cam = cv2.VideoCapture(self.cam_id)

		if not self.cam.isOpened():
			print("Video Source not found...")
			exit()

		self._label = label

		self.process_webcam()

	
	def process_webcam(self):
		ret, frame = self.cam.read()

		if not ret:
			print("failed to grab frame")
			exit()

		self.most_recent_capture_arr = frame


		img_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		
		self.most_recent_capture = Image.fromarray(img_)

		imgtk = ImageTk.PhotoImage(image=self.most_recent_capture)

		self._label.imgtk = imgtk
		self._label.config(image=imgtk)

		self._label.after(20, self.process_webcam)

	def register(self):
		try:
			DeepFace.extract_faces(self.most_recent_capture_arr)

			self.second_capture = self.most_recent_capture_arr
			self.second_capture = tf.image.stateless_random_brightness(self.second_capture, max_delta=0.02, seed=(1,2))
			self.second_capture = tf.image.stateless_random_contrast(self.second_capture, lower=0.6, upper=1, seed=(1,3))
			# self.second_capture = tf.image.stateless_random_crop(self.second_capture, size=(20,20,3), seed=(1,2))
			self.second_capture = tf.image.flip_left_right(self.second_capture)
			self.second_capture = tf.image.stateless_random_jpeg_quality(self.second_capture, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
			self.second_capture = tf.image.stateless_random_saturation(self.second_capture, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))

			DeepFace.extract_faces(self.second_capture.numpy())


		except ValueError:
			util.msg_box('Face could not be detected.', 'Face could not be detected. Please confirm that the picture is a face photo')
			return

		self.register_button.destroy()

		self.text_label_register = util.get_text_label(self.main_screen, 'Please, input user name:', (550, 100))

		self.entry_text_register = util.get_entry_text(self.main_screen, (550, 150))

		self.acpt_register_button = util.get_button(self.main_screen, "Accept", 'green', (550, 300), self.acpt_new_user, 'white')
		self.main_screen.bind('<Return>', self.temp)
		self.try_again_register_button = util.get_button(self.main_screen, "Try again", 'red', (550, 400), self.try_again_new_user, 'white')

		self.capture_label = util.get_img_label(self.main_screen, (950, 0), (500, 500))

		self.add_img_to_label(self.capture_label)
	
	def temp(self, event):
		if self.register_button.winfo_exists():
			self.register()
		elif self.acpt_register_button.winfo_exists():
			self.acpt_new_user()
	
	def add_img_to_label(self, label: tk.Label):
		imgtk = ImageTk.PhotoImage(image=self.most_recent_capture)
		label.imgtk = imgtk
		label.config(image=imgtk)

		self.register_capture = self.most_recent_capture_arr.copy()
	
	def acpt_new_user(self):
		name = self.entry_text_register.get(1.0, "end-1c")
		name = name.strip()
		name = name.replace("\n", "").replace("\r", "")
		name = name.lower()
		
		if name:
			is_write = True
			if os.path.exists(f'{self.db_dir}/{name}'):
				msg_box = tk.messagebox.askquestion('Register name already exist', 'Register name already exist, do you want to rewrite?',
                                        icon='warning')
				if msg_box != 'yes':
					self.entry_text_register.delete(1.0, 2.0)
					is_write = False
			else:
				os.makedirs(f'{self.db_dir}/{name}')

			if is_write:
				cv2.imwrite(os.path.join(self.db_dir, f'{name}/{name}.jpg'), self.register_capture)

				# _img = self.register_capture
				# _img = tf.image.stateless_random_brightness(_img, max_delta=0.02, seed=(1,2))
				# _img = tf.image.stateless_random_contrast(_img, lower=0.6, upper=1, seed=(1,3))
				# # _img = tf.image.stateless_random_crop(_img, size=(20,20,3), seed=(1,2))
				# _img = tf.image.flip_left_right(_img)
				# _img = tf.image.stateless_random_jpeg_quality(_img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
				# _img = tf.image.stateless_random_saturation(_img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))

				cv2.imwrite(os.path.join(self.db_dir, f'{name}/{name}2.jpg'), self.second_capture.numpy())


				util.msg_box('Success', 'Register Successfully!')
			
				self.reset()

		else:
			util.msg_box('Failed to read name', 'Please enter your name')

	def try_again_new_user(self):
		try:
			DeepFace.extract_faces(self.most_recent_capture_arr)
		except ValueError:
			util.msg_box('Face could not be detected.', 'Face could not be detected. Please confirm that the picture is a face photo')
			return
		self.add_img_to_label(self.capture_label)
	
	def start(self):
		self.main_screen.mainloop()


def main():
	app = Resigter(0)
	app.start()

if __name__ == "__main__":
	main()