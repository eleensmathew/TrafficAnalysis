import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from ultralytics import YOLO
import numpy as np
import pathlib
import matplotlib.pyplot as plt

import gradio as gr

def detect_all(img_path):
    original_image = cv2.imread(img_path)
    overcrowding_model= YOLO("best.pt")
    helmet_model= YOLO("best_helmet.pt")
    overcrowding_results= overcrowding_model(img_path)
    for det in overcrowding_results[0].boxes.xyxy:
    # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, det[:4])
        crop_img = original_image[y1:y2, x1:x2]
        helmet_results= helmet_model(crop_img)
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for det in helmet_results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, det[:4])
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite('combined_result.jpg', original_image)
def detect_overcrowding(img_path):
    model = YOLO("best.pt")
    #model.predict(source = img_path, conf=0.5, show = True, save=True, save_dir = img_path)
    overcrowding_results= model(img_path)
    print(overcrowding_results)
    return img_path

def detect_helmet(img_path):
    model = YOLO("best_helmet.pt")
    model.predict(source = img_path, conf=0.5, show = True, save=True, save_dir = img_path)
    
    return img_path

detect_all('/home/eleensmathew/TrafficAnalysis/current_frame.jpg')

#detect('/home/eleensmathew/TrafficAnalysis/current_frame.jpg')
# Define the Gradio interface
#iface = gr.Interface(fn=detect, inputs='file', outputs='file')

# Launch the interface
#iface.launch(share=True)
#detect('/home/eleensmathew/TrafficAnalysis/data/videos/video1.mp4')
#detect('/home/eleensmathew/TrafficAnalysis/data/videos/video2.mp4')
#detect('/home/eleensmathew/TrafficAnalysis/data/videos/video3.mp4')
#detect('/home/eleensmathew/TrafficAnalysis/data/videos/video4.mp4')
#detect('/home/eleensmathew/TrafficAnalysis/data/videos/video5.mp4')

#accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="0")
#