import cv2
import face_recognition

image_to_recognize_path = 'C:/Users/Abhishek/Face recognition using python/images/shubham2.jpg'

original_image = cv2.imread(image_to_recognize_path)

saurabh_image = face_recognition.load_image_file('C:/Users/Abhishek/Face recognition using python/samples/saurabh3.jpg')
saurabh_face_encodings = face_recognition.face_encodings(saurabh_image)[0]

obama_image = face_recognition.load_image_file('C:/Users/Abhishek/Face recognition using python/samples/obama.jpg')
obama_face_encodings = face_recognition.face_encodings(obama_image)[0]

known_face_encodings = [saurabh_face_encodings, obama_face_encodings]
known_face_names = ["Saurabh Upadhyay", "Unknown Face"]

image_to_recognize = face_recognition.load_image_file(image_to_recognize_path)

image_to_recognize_encodings = face_recognition.face_encodings(image_to_recognize)[0]

face_distances = face_recognition.face_distance(known_face_encodings, image_to_recognize_encodings)

for i,face_distance in enumerate(face_distances):
    print("The calculated face distance is {:.2} against the sample {}".format(face_distance, known_face_names[i]))
    
    print("The matching percentage is {} against the sample {}".format(round(((1-float(face_distance))*100),2), known_face_names[i]))