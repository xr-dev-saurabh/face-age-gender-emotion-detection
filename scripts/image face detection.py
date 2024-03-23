
import cv2
import face_recognition

image_to_detect = cv2.imread('C:/Users/Abhishek/Face recognition using python/images/shubham.jpg')

all_face_locations = face_recognition.face_locations(image_to_detect, model='hog')

print('There are {} no of faces in image.'.format(len(all_face_locations)))