from deepface import DeepFace


model = DeepFace.find('_temp.jpg', 'data/minh trinh', model_name='Facenet')

print(model)