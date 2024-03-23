import cv2
import face_recognition

original_image = cv2.imread('C:/Users/Abhishek/Face recognition using python/images/saurabh2.jpg')

saurabh_image = face_recognition.load_image_file('C:/Users/Abhishek/Face recognition using python/samples/saurabh3.jpg')
saurabh_face_encodings = face_recognition.face_encodings(saurabh_image)[0]

known_face_encodings = [saurabh_face_encodings]
known_face_names = ["Saurabh Upadhyay"]

image_to_recognize = face_recognition.load_image_file('C:/Users/Abhishek/Face recognition using python/images/saurabh2.jpg')

all_face_locations = face_recognition.face_locations(image_to_recognize, model='hog')
all_face_encodings = face_recognition.face_encodings(image_to_recognize, all_face_locations)

print('There are {} no of faces in image.'.format(len(all_face_locations)))

for current_face_location, current_face_encodings in zip(all_face_locations, all_face_encodings):
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encodings)
    name_of_person = 'Unknown Face'
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]
        
    cv2.rectangle(original_image,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
    
cv2.imshow("Face Identified ",original_image)