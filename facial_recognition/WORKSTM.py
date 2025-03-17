import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

yeshwanth_image = face_recognition.load_image_file("faces/yeshwanth.png")
yeshwanth_face_encoding = face_recognition.face_encodings(yeshwanth_image)[0]

mason_image = face_recognition.load_image_file("faces/mason.png")
mason_face_encoding = face_recognition.face_encodings(mason_image)[0]

hamsan_image = face_recognition.load_image_file("faces/hamsan.png")
hamsan_face_encoding = face_recognition.face_encodings(hamsan_image)[0]

vj_image = face_recognition.load_image_file("faces/vijay.png")
vijay_face_encoding = face_recognition.face_encodings(vj_image)[0]

dhan_image = face_recognition.load_image_file("faces/dhanush.png")
dhanush_face_encoding = face_recognition.face_encodings(dhan_image)[0]


known_face_encodings = [
    yeshwanth_face_encoding,
    mason_face_encoding,
    hamsan_face_encoding,
    vijay_face_encoding,
    dhanush_face_encoding
]
known_face_names = [
    "Yeshwanth",
    "Mason",
    "Hamsan",
    "Vijay",
    "Dhanush"
]

face_locations = []
face_encodings = []
face_names = []

while True:
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

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Something', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()