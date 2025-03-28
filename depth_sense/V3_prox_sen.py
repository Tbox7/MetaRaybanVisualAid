import argparse
import cv2
import numpy as np
import os
import torch
import winsound
from depth_anything_v2.dpt import DepthAnythingV2

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

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access webcam.")
        exit()

    print("Press 'q' to exit the application.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break

            # Inference
            depth = depth_anything.infer_image(frame, args.input_size)
            norm_depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth_uint8 = (norm_depth * 255.0).astype(np.uint8)

            # Get depth at the center
            h, w = depth.shape
            center_depth_value = norm_depth[h // 2, w // 2]
            approx_distance_cm = (1.0 - center_depth_value) * 200  # Arbitrary scale

            # Beep if within ~1 foot (30 cm)
            if approx_distance_cm <= 30:
                winsound.Beep(1000, 150)  # frequency, duration in ms

            # Optional: draw a center circle
            cv2.circle(depth_uint8, (w // 2, h // 2), 5, (255, 255, 255), -1)

            # Show depth map
            cv2.imshow("Live Depth Map (Grayscale)", depth_uint8)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()