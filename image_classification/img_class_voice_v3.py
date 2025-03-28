import torch
import clip
import cv2
import pyttsx3  # Text-to-speech
import speech_recognition as sr  # Speech recognition
import threading
import time
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# Initialize text-to-speech engine
engine = pyttsx3.init()
stop_flag = False
def speak(text):
    engine.say(text)
    engine.runAndWait()
# Initialize speech recognition
recognizer = sr.Recognizer()
spoken_text = ""
def listen_for_trigger():
    global spoken_text, stop_flag
    while not stop_flag:  # Run until stop_flag is set to True
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                print("Listening for trigger word...")
                audio = recognizer.listen(source, timeout=3)
                spoken_text = recognizer.recognize_google(audio).lower()
            except sr.UnknownValueError:
                spoken_text = ""
            except sr.RequestError:
                print("Speech recognition service unavailable")
                spoken_text = ""
# Start speech recognition in a separate thread
thread = threading.Thread(target=listen_for_trigger, daemon=True)
thread.start()
# Object categories
object_labels = ["keyboard", "car", "bottle", "chair", "laptop", "phone", "tree", 
                 "book", "table", "toothbrush", "toothpaste", "pencil","bicycle", 
                 "cup","door","glass","computer monitor"]
text_inputs = clip.tokenize(object_labels).to(device)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)  
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
trigger_word = "identify"
last_label = None
last_confidence = 0
last_frame_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    last_frame_time = time.time()
    height, width, _ = frame.shape
    box_width, box_height = int(width * 0.40), int(height * 0.90)
    x1, y1 = (width - box_width) // 2, (height - box_height) // 2
    x2, y2 = x1 + box_width, y1 + box_height
    sub_frame = frame[y1:y2, x1:x2]
    sub_frame = cv2.resize(sub_frame, (256, 256))  
    image = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        similarity = (image_features @ text_features.T).softmax(dim=-1)
    best_match_idx = similarity.argmax().item()
    best_match_label = object_labels[best_match_idx]
    confidence = similarity[0, best_match_idx].item()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    if confidence > 0.3:
        label_text = f"{best_match_label} ({confidence:.2f})"
        label_x, label_y = x1, y2 + 30
        cv2.rectangle(frame, (label_x, label_y - 25), (label_x + 250, label_y), (0, 0, 255), -1)
        cv2.putText(frame, label_text, (label_x + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
# Check if trigger word was spoken
    if trigger_word in spoken_text:
        print(f"Trigger word detected: {spoken_text}")
        speak(f"I see a {best_match_label}")
        spoken_text = ""  # Reset trigger word detection
    cv2.imshow("CqLIP Object Detection (Optimized)", frame)
# Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
stop_flag = True  
thread.join()
cap.release()
cv2.destroyAllWindows()