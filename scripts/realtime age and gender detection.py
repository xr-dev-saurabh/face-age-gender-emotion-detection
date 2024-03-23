import cv2
import face_recognition

webcam_video_stream = cv2.VideoCapture(0)

all_face_locations = []
while True:
    ret,current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    all_face_locations = face_recognition.face_locations(current_frame_small,model='hog')
    for index,current_face_location in enumerate(all_face_locations):
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
       
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
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,225),2)
            
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, gender+" "+age+"yrs", (left_pos,bottom_pos+20), font, 0.5, (0,255,0),1)
    cv2.imshow("Video Webcam",current_frame)
         
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam_video_stream.release()
cv2.destroyAllWindows()
 
