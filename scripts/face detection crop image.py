import cv2
import face_recognition

image_to_detect = cv2.imread('C:/Users/Abhishek/Face recognition using python/images/tarang.jpg')

all_face_locations = face_recognition.face_locations(image_to_detect, model='hog')

print('There are {} no of faces in image.'.format(len(all_face_locations)))

for index,current_face_location in enumerate(all_face_locations):
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
    current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
    cv2.imshow("Face no "+str(index+1),current_face_image)