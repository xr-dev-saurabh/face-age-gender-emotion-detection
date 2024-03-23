#import libraries
import cv2
import face_recognition

#using webcam to detect face
webcam_video_stream = cv2.VideoCapture(0)

#loading and creating face encodings
saurabh_image = face_recognition.load_image_file('C:/Users/Abhishek/Face recognition using python/samples/saurabh3.jpg')
saurabh_face_encodings = face_recognition.face_encodings(saurabh_image)[0]

#storing encodings and names in array
known_face_encodings = [saurabh_face_encodings]
known_face_names = ["Saurabh Upadhyay"]

#declaring empty array for storage
all_face_locations = []
all_face_encodings = []
all_face_names = []

#looping to detect face from webcam
while True:
    ret,current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    all_face_locations = face_recognition.face_locations(current_frame_small,model='hog')
    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)

    #looping to compare known face encodings to newly detected face 
    for current_face_location, current_face_encodings in zip(all_face_locations, all_face_encodings):
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encodings)
        name_of_person = 'Unknown Face'
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
         
            #to draw rectange and names below it
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
        
    cv2.imshow("Face Identified ",current_frame)
    
   
     #press q to exit from loop    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam_video_stream.release()
cv2.destroyAllWindows()
 
