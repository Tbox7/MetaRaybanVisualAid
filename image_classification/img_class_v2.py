import torch
import clip
import cv2
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
object_labels = ["keyboard", "car", "bottle", "chair", "laptop", "phone", "tree", 
                 "book", "table", "toothbrush", "toothpaste", "pencil","bicycle", 
                 "cup","door","glass","computer monitor"]
text_inputs = clip.tokenize(object_labels).to(device)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
frame_count = 0  
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    height, width, _ = frame.shape
    box_width, box_height = int(width * 0.4), int(height * 0.8)
    x1, y1 = (width - box_width) // 2, (height - box_height) // 2
    x2, y2 = x1 + box_width, y1 + box_height
    sub_frame = frame[y1:y2, x1:x2]
    sub_frame = cv2.resize(sub_frame, (224, 224)) 
    image = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    # Preprocess image for CLIP
    image_input = preprocess(image).unsqueeze(0).to(device)
    if frame_count % 2 == 0:
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            similarity = (image_features @ text_features.T).softmax(dim=-1)
        best_match_idx = similarity.argmax().item()
        best_match_label = object_labels[best_match_idx]
        confidence = similarity[0, best_match_idx].item()
    frame_count += 1  
# Draw bounding box and label
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    if confidence > 0.4:
        label_text = f"{best_match_label} ({confidence:.2f})"
        label_x, label_y = x1, y2 + 30
        cv2.rectangle(frame, (label_x, label_y - 25), (label_x + 250, label_y), (0, 0, 255), -1)
        cv2.putText(frame, label_text, (label_x + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("CLIP Centered Object Detection", frame)
#press"q" to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
