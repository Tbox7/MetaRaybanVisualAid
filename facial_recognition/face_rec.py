import face_recognition
import cv2
import numpy as np
video_capture = cv2.VideoCapture(0)
# Load images of known individuals and generate their face encodings
# These encodings are numerical representations of facial features
ryan_image = face_recognition.load_image_file("faces/ryan.png")
ryan_face_encoding = face_recognition.face_encodings(ryan_image)[0]
martha_image = face_recognition.load_image_file("faces/martha.png")
martha_face_encoding = face_recognition.face_encodings(martha_image)[0]
dave_image = face_recognition.load_image_file("faces/dave.png")
dave_face_encoding = face_recognition.face_encodings(dave_image)[0]
vijay_image = face_recognition.load_image_file("faces/vijay.png")
vijay_face_encoding = face_recognition.face_encodings(vijay_image)[0]
# Store the face encodings and corresponding names in lists
known_face_encodings = [
    ryan_face_encoding,
    martha_face_encoding,
    dave_face_encoding,
    vijay_face_encoding
]
known_face_names = [
    "Ryan_reynolds",  
    "Martha_stewart",  
    "Dave_chappelle",  
    "Vijay"            
]
# Initialize variables to store face detection results
face_locations = []  
face_encodings = []  
face_names = []     

while True:
    # Capture a single frame from the video feed
    ret, frame = video_capture.read()
    if not ret:
        
        print("No frame")
        break
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    rgb_small_frame = rgb_small_frame.astype(np.uint8)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown" 
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    # Draw rectangles and labels around detected faces
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
#live video feed press 'q' to exit
    cv2.imshow('Something', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
