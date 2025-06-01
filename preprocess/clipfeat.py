import os
import torch
import clip
import cv2
import numpy as np
from PIL import Image

def extract_features(video_path, output_path, model, preprocess, device, L=10):
    """Extracts CLIP features from a video, divides them into L intervals, and averages each interval."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_features = []
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_input = preprocess(image).unsqueeze(0).to(device)
            image_features = model.encode_image(image_input)
            frame_features.append(image_features.cpu())

    cap.release()

    if frame_features:
        frame_features = torch.cat(frame_features, dim=0)
        num_frames, dC = frame_features.shape
        interval_size = max(1, num_frames // L)

        # Divide into L intervals and compute average per interval
        dance_feature = torch.stack([frame_features[i * interval_size:(i + 1) * interval_size].mean(dim=0)
                                     for i in range(L)])

        torch.save(dance_feature, output_path)
        print(f"Saved features: {output_path} ({dance_feature.shape})")
    else:
        print(f"No frames extracted from {video_path}")

def process_video_folder(input_folder, output_folder):
    """Processes all video files in a folder and saves processed dance features in the output folder."""
    os.makedirs(output_folder, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_features.pt")
            extract_features(video_path, output_path, model, preprocess, device, L=10)

def main():
    input_folder = r"/home/sangheon/Desktop/BeatDance/data/dance_video"
    output_folder = r"/home/sangheon/Desktop/BeatDance/data/video_feature"
    process_video_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()