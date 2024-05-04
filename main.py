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
    resized_image=cv2.resize(original_image, (640, 640))
    overcrowding_model= YOLO("best.pt")
    helmet_model= YOLO("best_helmet.pt")
    overcrowding_results= overcrowding_model(resized_image)
    class_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # replace with your actual class colors

    
    for det in overcrowding_results[0].boxes.xyxy:

        #print(det)
        x1, y1, x2, y2 = map(int, det[:4])
        helmet_results= helmet_model(resized_image)
        cv2.rectangle(resized_image, (x1, y1), (x2, y2), class_colors[2], 2)
        label = overcrowding_results[0].names[0]
        print(label)
        cv2.putText(resized_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, class_colors[2], 2)

    class_labels = ['With Helmet', 'Without Helmet']
    for det in helmet_results[0].boxes:
        det1=det.xyxy.flatten()
        x1, y1, x2, y2 = map(int, det1[:4].tolist())
        cls_value = int(det.cls.item())
        cv2.rectangle(resized_image, (x1, y1), (x2, y2), class_colors[cls_value], 2)
        print("cls",cls_value)
        print("check")
        cv2.putText(resized_image, class_labels[cls_value], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, class_colors[cls_value], 2)
    
    cv2.imwrite('combined_result.jpg', resized_image)
    return 'combined_result.jpg'



#detect_all('/home/eleensmathew/TrafficAnalysis/archive (1)/images/BikesHelmets731.png')


iface = gr.Interface(fn=detect_all, inputs='file', outputs='image')

iface.launch(share=True)

#accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="0")
#