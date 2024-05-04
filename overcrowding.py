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

def detect_overcrowding(img_path):
    model = YOLO("best.pt")
    model.predict(source = img_path, conf=0.5, show = True, save=True, save_dir = img_path)
    #overcrowding_results= model(img_path)
    #print(overcrowding_results)
    return img_path


iface = gr.Interface(fn=detect_overcrowding, inputs='file', outputs='file')
iface.launch(share=True)
#detect_overcrowding('/home/eleensmathew/TrafficAnalysis/data/videos/video1.mp4')
#detect_overcrowding('/home/eleensmathew/TrafficAnalysis/data/videos/video2.mp4')
#detect_overcrowding('/home/eleensmathew/TrafficAnalysis/data/videos/video3.mp4')
#detect_overcrowding('/home/eleensmathew/TrafficAnalysis/data/videos/video4.mp4')
#detect_overcrowding('/home/eleensmathew/TrafficAnalysis/data/videos/video5.mp4')

#accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="0")
#