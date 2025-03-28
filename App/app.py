import cv2
import torch
import clip
import face_recognition
import pyttsx3
import speech_recognition as sr
import threading
import numpy as np
from PIL import Image
from assistant import AzureVision, AzureSpeech
import argparse
import winsound
from depth_anything_v2.dpt import DepthAnythingV2
import os
from dotenv import load_dotenv

load_dotenv()

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY")
AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

vision = AzureVision(AZURE_VISION_KEY, AZURE_VISION_ENDPOINT)
speech = AzureSpeech(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION)

# Load CLIP model for image classification
model, preprocess = clip.load("ViT-B/32", device=device)
object_labels = ["keyboard", "car", "bottle", "phone", "book", "screwdriver", "mouse", "glass"]
text_inputs = clip.tokenize(object_labels).to(device)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize speech recognition
recognizer = sr.Recognizer()
spoken_text = ""

# Load known faces
face_encodings = []
face_names = []
known_faces = {"Ryan Reynolds": "faces/ryan.png", "Martha stewart": "faces/Martha.png", "Dave Chappelle": "faces/dave.png", "Vijay": "faces/vijay.png"}

for name, filepath in known_faces.items():
    image = face_recognition.load_image_file(filepath)
    encoding = face_recognition.face_encodings(image)[0]
    face_encodings.append(encoding)
    face_names.append(name)

stop_flag = False
def speak(text):
    print(text)  # Print the text to the terminal
    engine.say(text)
    engine.runAndWait()

def listen_for_trigger():
    global spoken_text, stop_flag
    while not stop_flag:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=100)
                spoken_text = recognizer.recognize_google(audio).lower()
            except sr.UnknownValueError:
                spoken_text = ""
            except sr.RequestError:
                print("Speech recognition service unavailable")
                spoken_text = ""

# Start trigger listening thread
thread = threading.Thread(target=listen_for_trigger, daemon=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 - Live Webcam Feed')
    parser.add_argument('--input-size', type=int, default=518, help='Input size for the depth model')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'], help='Model encoder type')
    args = parser.parse_args()

    # Set device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # Load the Depth Anything V2 model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    checkpoint_path = rf"C:\Users\Reach\python\depth_sense_final\Depth-Anything-V2\checkpoints\depth_anything_v2_{args.encoder}.pth"
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
thread.start()

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

trigger_word = "identify"
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame for processing
    rgb_frame = frame[:, :, ::-1]
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings_in_frame = face_recognition.face_encodings(small_frame, face_locations)

    detected_names = []
    for encoding in face_encodings_in_frame:
        matches = face_recognition.compare_faces(face_encodings, encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(face_encodings, encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = face_names[best_match_index]
        detected_names.append(name)

    # Image Classification (only if 'r' is not pressed)
    key = cv2.waitKey(1) & 0xFF
    if key != ord('r'):  # Skip object classification if 'r' is pressed
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_input = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            similarity = (image_features @ text_features.T).softmax(dim=-1)
        best_match_idx = similarity.argmax().item()
        best_match_label = object_labels[best_match_idx]
        confidence = similarity[0, best_match_idx].item()
        depth = depth_anything.infer_image(frame, args.input_size)
        norm_depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth_uint8 = (norm_depth * 255.0).astype(np.uint8)
        h, w = depth.shape
        center_depth_value = norm_depth[h // 2, w // 2]
        approx_distance_cm = (1.0 - center_depth_value) * 200
        if approx_distance_cm <= 30:
            winsound.Beep(1000, 150)
        # Provide feedback based on priority
        if trigger_word in spoken_text:
            speak(f"I see a {best_match_label}")
            spoken_text = ""

        # Display object classification on the frame
        cv2.putText(frame, best_match_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display frame
    for (top, right, bottom, left), name in zip(face_locations, detected_names):
        top, right, bottom, left = top*4, right*4, bottom*4, left*4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow("Integrated AI Assistance", frame)
    cv2.imshow("Live Depth Map (Grayscale)", depth_uint8)
    # Key press actions
    if key == ord('f'):
        if detected_names:
            speak(f"I see {', '.join(detected_names)}")
        else:
            speak("No faces detected")
    if key == ord('r'):
        text = vision.extract_text(frame)
        speech.speak(text if text else "No text detected.")
    if key == ord('q'):
        break

stop_flag = True
thread.join()
cap.release()
cv2.destroyAllWindows()
