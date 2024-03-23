import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import face_recognition

image_to_detect = cv2.imread('C:/Users/Abhishek/Face recognition using python/images/shubham.jpg')
resize_window = cv2.resize(image_to_detect, (1200, 1200))

face_exp_model = model_from_json(open("C:/Users/Abhishek/Face recognition using python/dataset/facial_expression_model_structure.json","r").read())
face_exp_model.load_weights('C:/Users/Abhishek/Face recognition using python/dataset/facial_expression_model_weights.h5')
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

all_face_locations = face_recognition.face_locations(image_to_detect, model='hog')

print('There are {} no of faces in image.'.format(len(all_face_locations)))

for index,current_face_location in enumerate(all_face_locations):
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    
    cv2.rectangle(image_to_detect,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,225),2)
    
    current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY) 
    
    current_face_image = cv2.resize(current_face_image, (48, 48))
    
    img_pixels = image.img_to_array(current_face_image)
    
    img_pixels = np.expand_dims(img_pixels, axis = 0)
   
    img_pixels /= 255 
    
    exp_predictions = face_exp_model.predict(img_pixels) 
   
    max_index = np.argmax(exp_predictions[0])
    
    emotion_label = emotions_label[max_index]
    
    
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect, emotion_label, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
   
cv2.imshow('Image face emotions',resize_window)