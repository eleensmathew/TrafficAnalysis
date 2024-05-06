
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img,(kernel_size, kernel_size), 0)
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2 :
        channel_count = img.shape[2]
        ignore_mask = (255,) * channel_count
    else:
        ignore_mask = 255
        
    cv2.fillPoly(mask,vertices, ignore_mask)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def draw_lines(image,img, lines, color=[0, 0, 255], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_lanes_lines = []
    right_lanes_lines = []
    mask = np.zeros_like(img)

    #cv2.imwrite('/home/eleensmathew/TrafficAnalysis/output_frame/123.png', img_dup)

    ##The average is easier to compute in slope intercept form.
    ## The average of slopes and average of intercepts of all lines is
    ## a good representation of the line.
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                if x2 == x1:
                    x2 = x1 + 0.1  # or x1 = x2 + 0.1, depending on your needs

                slope = (y2-y1)/(x2-x1)
                if slope < -0.4 and slope > -0.9:
                    left_lanes_lines.append((x1, y1, x2, y2))
                if (slope > 0.4 and slope <.9):
                    right_lanes_lines.append((x1, y1, x2, y2))

    left_lane_detection = [sum(y) / len(y) for y in zip(*left_lanes_lines)]
    right_lane_detection = [sum(y) / len(y) for y in zip(*right_lanes_lines)]
    
    x2_left=0
    x1_left=0
    y1_left=0;
    y2_left=0
    

    if left_lane_detection is not None and len(left_lane_detection) > 0:
        slope = (left_lane_detection[3]-left_lane_detection[1])/ (left_lane_detection[2]-left_lane_detection[0])
        intercept = left_lane_detection[1] - slope*left_lane_detection[0]

        y1_left = img.shape[0]  # bottom of the image
        y2_left = int(y1_left * 0.58) #top of line, similar to mask height
        x1_left = int((y1_left - intercept) / slope)
        x2_left = int((y2_left - intercept) / slope)

        cv2.line(img, (x1_left, y1_left), (x2_left, y2_left), color, thickness)
        ##cv2.line(img, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), color, thickness)
    x2_right=0
    x1_right=0
    y1_right=0
    y2_right=0

    if right_lane_detection is not None and len(right_lane_detection) > 0:
        slope = (right_lane_detection[3] - right_lane_detection[1]) / (right_lane_detection[2] - right_lane_detection[0])
        intercept = right_lane_detection[1] - slope * right_lane_detection[0]

        y1_right = img.shape[0]  # bottom of the image
        y2_right = int(y1_right  * 0.58)
        x1_right = int((y1_right - intercept) / slope)
        x2_right = int((y2_right - intercept) / slope)
        cv2.line(img, (x1_right, y1_right), (x2_right, y2_right), color, thickness)
    if left_lane_detection is not None and len(left_lane_detection) > 0 and right_lane_detection is not None and len(right_lane_detection) > 0:
        # Draw a line between the top points of the left and right lanes
        cv2.line(img, (x2_left, y2_left), (x2_right, y2_right), color, thickness)
        cv2.line(img, (x1_left, y1_left), (x1_right, y1_right), color, thickness)
        roi_corners = np.array([[(x1_left, y1_left), (x2_left, y2_left), (x2_right, y2_right), (x1_right, y1_right)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_corners, (255, 255, 255))
        
        img = cv2.bitwise_and(image, mask)
        
    
    return img

def hough_lines(image,img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    #cv2.imwrite('/home/eleensmathew/TrafficAnalysis/output_frame/ppp.png', img)
    line_img=draw_lines(image,line_img, lines)
    return line_img
def weighted_img(img, intial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(intial_img, α, img, β, λ)


#for i, img in enumerate(os.listdir("C:/Users/Sharath/Desktop/SDND-Term-1/test_images")):

def read():
    for i, img in enumerate(os.listdir("/home/eleensmathew/TrafficAnalysis/data/frames")):
        image = mpimg.imread('/home/eleensmathew/TrafficAnalysis/data/frames/' + img)
        orig_image=image

        gray_image = grayscale(image)        
        kernel_size = 11
        blur_gray = gaussian_blur(image,kernel_size)
        low_threshold = 50
        high_threshold = 150
        edges = canny(blur_gray, low_threshold, high_threshold)
        
        
        vertices = np.array([[(0,540),(460.79999, 313.2),
                            (499.2, 313.2 ), (960, 540)]], dtype=np.int32)
        
        # These Values are based ion trail and error to choose the perfect apex and vertices of the triangle i.e Region of Interest
        masked_edges = region_of_interest(edges,vertices)
        
            
        # Define the Hough transform parameters
        rho = 1
        theta = np.pi/180
        threshold = 20
        min_line_length = 40
        max_line_gap = 300

        # Run Hough on edge detected image
        lines = hough_lines(image,masked_edges, rho, theta, threshold,
                                min_line_length, max_line_gap)

        # Draw the lines on the edge image
        #result = weighted_img(lines, image)
        cv2.imwrite('/home/eleensmathew/TrafficAnalysis/output_frame/out'+img, lines)
        plt.figure()
        plt.imshow(lines)

def extract_frames(video_path, output_path, frame_interval=60):
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    success = True

    while success:
        success, image = vidcap.read()
        if count % frame_interval == 0:
            cv2.imwrite(output_path + "/frame%d.jpg" % count, image)
        count += 1

# Usage
def detect_motion(frame1, frame2):
    # Convert the frames to grayscale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the two frames
    frame_diff = cv2.absdiff(frame1_gray, frame2_gray)

    # Apply a threshold to the difference (adjust the second parameter to increase or decrease sensitivity)
    _, threshold = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)

    # Apply a series of dilations to fill in the holes
    dilated = cv2.dilate(threshold, None, iterations=5)

    # Find contours of the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours: ", len(contours))
    cv2.imshow('Dilated', dilated)
    cv2.waitKey(6000)
    cv2.destroyAllWindows()
    # If contours are detected, there is motion
    if len(contours) > 2:
        return True
    else:
        return False
def get_frames():
    image_files = os.listdir('/home/eleensmathew/TrafficAnalysis/output_frame/')
    image_files.sort()

    # Read the first image outside the loop
    prev_frame = cv2.imread(os.path.join('/home/eleensmathew/TrafficAnalysis/output_frame/', image_files[0]))

    # Iterate over the rest of the image files
    for image_file in image_files[1:]:
        # Read the next frame
        curr_frame = cv2.imread(os.path.join('/home/eleensmathew/TrafficAnalysis/output_frame/', image_file))

        # Call detect_motion with the previous and current frame
        print(detect_motion(prev_frame, curr_frame))

        # Update prev_frame to the current frame for the next iteration
        prev_frame = curr_frame
# Load the image

extract_frames('/home/eleensmathew/TrafficAnalysis/data/videos/solidYellowLeft (1).mp4', '/home/eleensmathew/TrafficAnalysis/data/frames/', frame_interval=60)
read()
get_frames()