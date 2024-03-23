import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import face_recognition

saurabh_image = face_recognition.load_image_file('C:/Users/Abhishek/Face recognition using python/samples/saurabh3.jpg')
saurabh_face_encodings = face_recognition.face_encodings(saurabh_image)[0]

known_face_encodings = [saurabh_face_encodings]
known_face_names = ["Saurabh Upadhyay"]

image_to_recognize = face_recognition.load_image_file('C:/Users/Abhishek/Face recognition using python/images/saurabh2.jpg')

image_to_detect = cv2.imread('C:/Users/Abhishek/Face recognition using python/images/saurabh2.jpg')

face_exp_model = model_from_json(open("C:/Users/Abhishek/Face recognition using python/dataset/facial_expression_model_structure.json","r").read())
face_exp_model.load_weights('C:/Users/Abhishek/Face recognition using python/dataset/facial_expression_model_weights.h5')
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

all_face_locations = face_recognition.face_locations(image_to_detect, model='hog')
all_face_encodings = face_recognition.face_encodings(image_to_recognize, all_face_locations)

print('There are {} no of faces in image.'.format(len(all_face_locations)))

for current_face_location, current_face_encodings in zip(all_face_locations, all_face_encodings):
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encodings)
    name_of_person = 'Unknown Face'
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]
        
    

    for index,current_face_location in enumerate(all_face_locations):
        
        print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
        AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)
        gender_label_list = ['Male', 'Female']
        #declaring the file paths
        gender_protext = "C:/Users/Abhishek/Face recognition using python/dataset/gender_deploy.prototxt"
        gender_caffemodel = "C:/Users/Abhishek/Face recognition using python/dataset/gender_net.caffemodel"
        #creating the model
        gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
        #giving input to the model
        gender_cov_net.setInput(current_face_image_blob)
        #get the predictions from the model
        gender_predictions = gender_cov_net.forward()
        #find the max value of predictions index
        #pass index to label array and get the label text
        gender = gender_label_list[gender_predictions[0].argmax()]
        
        # Predicting Age
        #declaring the labels
        age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        #declaring the file paths
        age_protext = "C:/Users/Abhishek/Face recognition using python/dataset/age_deploy.prototxt"
        age_caffemodel = "C:/Users/Abhishek/Face recognition using python/dataset/age_net.caffemodel"
        #creating the model
        age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
        #giving input to the model
        age_cov_net.setInput(current_face_image_blob)
        #get the predictions from the model
        age_predictions = age_cov_net.forward()
        #find the max value of predictions index
        #pass index to label array and get the label text
        age = age_label_list[age_predictions[0].argmax()]
        cv2.rectangle(image_to_detect,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,225),2)
        
        current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY) 
        
        current_face_image = cv2.resize(current_face_image, (48, 48))
        
        img_pixels = image.img_to_array(current_face_image)
        
        img_pixels = np.expand_dims(img_pixels, axis = 0)
       
        img_pixels /= 255 
        
        exp_predictions = face_exp_model.predict(img_pixels) 
       
        max_index = np.argmax(exp_predictions[0])
        
        emotion_label = emotions_label[max_index]
        
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image_to_detect, name_of_person, gender+" "+age+"yrs"+" "+emotion_label, (left_pos,bottom_pos+20), font, 0.5, (0,255,0),1)
cv2.imshow("Age and Gender",image_to_detect)