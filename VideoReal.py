import cv2
import numpy as np
from tensorflow import keras
from helmet_detection import check

# Load your pre-trained CNN model
#model = keras.models.load_model('helmet_detection_model.h5')

def run_video(path):
    cap = cv2.VideoCapture(path)

    while True:
        # Read frame from video stream
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame if necessary (e.g., resizing, normalization)
        # Ensure that the input size matches the input size expected by your model
        
        # Predict helmet presence using your model
        # You may need to convert the frame to the appropriate format expected by your model
        #prediction = model.predict(np.expand_dims(frame, axis=0))
        cv2.imwrite('current_frame.jpg', frame)
        helmet_present = check('current_frame.jpg')
        if helmet_present =="Helmet":
            
            cv2.putText(frame, 'Helmet detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Helmet Not detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Helmet Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Process prediction results
        # For example, draw bounding boxes around detected helmets
        # You can customize this based on your model's output format
        # if prediction[0] == 1:  # Assuming 1 indicates helmet presence
        #     # Draw a bounding box around the helmet
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # # Display the frame with predictions
        # cv2.imshow('Helmet Detection', frame)
        
        # # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release the video stream and close all windows
    cap.release()
    cv2.destroyAllWindows()
run_video('/home/eleensmathew/TrafficAnalysis/data/videos/video1.mp4')
run_video('/home/eleensmathew/TrafficAnalysis/data/videos/video2.mp4')
run_video('/home/eleensmathew/TrafficAnalysis/data/videos/video3.mp4')
run_video('/home/eleensmathew/TrafficAnalysis/data/videos/video4.mp4')
run_video('/home/eleensmathew/TrafficAnalysis/data/videos/video5.mp4')