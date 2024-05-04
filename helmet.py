import gradio as gr
from ultralytics import YOLO

def detect_helmet(img_path):
    model = YOLO("best_helmet.pt")
    model.predict(source = img_path, conf=0.5, show = True, save=True, save_dir = img_path)
    #helmet_results= model(img_path)
    #print(helmet_results)
    return img_path


iface = gr.Interface(fn=detect_helmet, inputs='file', outputs='file')
iface.launch(share=True)

#detect_helmet('/home/eleensmathew/TrafficAnalysis/data/videos/video1.mp4')
#detect_helmet('/home/eleensmathew/TrafficAnalysis/data/videos/video2.mp4')
#detect_helmet('/home/eleensmathew/TrafficAnalysis/data/videos/video3.mp4')
#detect_helmet('/home/eleensmathew/TrafficAnalysis/data/videos/video4.mp4')
#detect_helmet('/home/eleensmathew/TrafficAnalysis/data/videos/video5.mp4')

#accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="0")
#