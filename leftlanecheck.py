
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


drawing = False
ix, iy = -1, -1
vertices = []
def get_vertices(img_path):
    '''draw vertices from bottom-left, top-left, top-right, bottom-right.'''
    global vertices
    def draw_roi(event, x, y, flags, param):
        global ix, iy, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            vertices.append((x, y))

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            
            # cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            # vertices.append((x, y))
        print(vertices)


    img = cv2.imread(img_path)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_roi)

    while(1):
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to exit
            break

    cv2.destroyAllWindows()

    vertices = np.array([vertices], dtype=np.int32)
    return vertices
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

def draw_polygon(image, img, vertices, color=[0, 0, 255], thickness=10):
    mask = np.zeros_like(img)
    vertices = [tuple(point) for point in vertices[0]]
    if vertices is not None and len(vertices) > 0:
        # Convert vertices to numpy array
        roi_corners = np.array([vertices], dtype=np.int32)
        #cv2.fillPoly(mask, roi_corners, (255, 255, 255))
        # Draw lines
        for i in range(len(vertices)):
            cv2.line(img, vertices[i], vertices[(i+1)%len(vertices)], color, thickness)
            #print(vertices[i],vertices[(i+1)%len(vertices)])
            #cv2.line(img, (x1_left, y1_left), (x2_left, y2_left), color, thickness)

        # Fill polygon
        
        cv2.fillPoly(mask, roi_corners, (255, 255, 255))
        img = cv2.bitwise_and(image, mask)
    

        # Apply mask
        #img = cv2.bitwise_and(image, mask)
        cv2.imwrite('/home/eleensmathew/TrafficAnalysis/output_frame/ppp.png', img)

    return img

def draw_lines(image, img, lines, color=[0, 0, 255], thickness=10):
    leftmost_line = None
    rightmost_line = None
    mask = np.zeros_like(img)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if leftmost_line is None or min(x1, x2) < min(leftmost_line[0], leftmost_line[2]):
                    leftmost_line = (x1, y1, x2, y2)
                if rightmost_line is None or max(x1, x2) > max(rightmost_line[0], rightmost_line[2]):
                    rightmost_line = (x1, y1, x2, y2)

    if leftmost_line is not None:
        cv2.line(img, (leftmost_line[0], leftmost_line[1]), (leftmost_line[2], leftmost_line[3]), color, thickness)

    if rightmost_line is not None:
        cv2.line(img, (rightmost_line[0], rightmost_line[1]), (rightmost_line[2], rightmost_line[3]), color, thickness)

    if leftmost_line is not None and rightmost_line is not None:
        # Draw a line between the top points of the left and right lanes
        cv2.line(img, (leftmost_line[2], leftmost_line[3]), (rightmost_line[2], rightmost_line[3]), color, thickness)
        cv2.line(img, (leftmost_line[0], leftmost_line[1]), (rightmost_line[0], rightmost_line[1]), color, thickness)
        roi_corners = np.array([[(leftmost_line[0], leftmost_line[1]), (leftmost_line[2], leftmost_line[3]), (rightmost_line[2], rightmost_line[3]), (rightmost_line[0], rightmost_line[1])]], dtype=np.int32)
        cv2.fillPoly(mask, roi_corners, (255, 255, 255))
        
        img = cv2.bitwise_and(image, mask)

    return img
# def draw_lines(image,img, lines, color=[0, 0, 255], thickness=10):
    
#     left_lanes_lines = []
#     right_lanes_lines = []
#     mask = np.zeros_like(img)
#     if lines is not None:
#         for line in lines:
#             for x1,y1,x2,y2 in line:
#                 if x2 == x1:
#                     x2 = x1 + 0.1  # or x1 = x2 + 0.1, depending on your needs

#                 slope = (y2-y1)/(x2-x1)
#                 if slope < 0:
#                     left_lanes_lines.append((x1, y1, x2, y2))
#                 if (slope >= 0):
#                     right_lanes_lines.append((x1, y1, x2, y2))

#     left_lane_detection = [sum(y) / len(y) for y in zip(*left_lanes_lines)]
#     right_lane_detection = [sum(y) / len(y) for y in zip(*right_lanes_lines)]
    
#     x2_left=0
#     x1_left=0
#     y1_left=0;
#     y2_left=0
    

#     if left_lane_detection is not None and len(left_lane_detection) > 0:
#         slope = (left_lane_detection[3]-left_lane_detection[1])/ (left_lane_detection[2]-left_lane_detection[0])
#         intercept = left_lane_detection[1] - slope*left_lane_detection[0]

#         y1_left = img.shape[0]  # bottom of the image
#         y2_left = int(y1_left * 0.58) #top of line, similar to mask height
#         x1_left = int((y1_left - intercept) / slope)
#         x2_left = int((y2_left - intercept) / slope)

#         cv2.line(img, (x1_left, y1_left), (x2_left, y2_left), color, thickness)
#         ##cv2.line(img, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), color, thickness)
#     x2_right=0
#     x1_right=0
#     y1_right=0
#     y2_right=0

#     if right_lane_detection is not None and len(right_lane_detection) > 0:
#         slope = (right_lane_detection[3] - right_lane_detection[1]) / (right_lane_detection[2] - right_lane_detection[0])
#         intercept = right_lane_detection[1] - slope * right_lane_detection[0]

#         y1_right = img.shape[0]  # bottom of the image
#         y2_right = int(y1_right  * 0.58)
#         x1_right = int((y1_right - intercept) / slope)
#         x2_right = int((y2_right - intercept) / slope)
#         cv2.line(img, (x1_right, y1_right), (x2_right, y2_right), color, thickness)
#     if left_lane_detection is not None and len(left_lane_detection) > 0 and right_lane_detection is not None and len(right_lane_detection) > 0:
#         # Draw a line between the top points of the left and right lanes
#         cv2.line(img, (x2_left, y2_left), (x2_right, y2_right), color, thickness)
#         cv2.line(img, (x1_left, y1_left), (x1_right, y1_right), color, thickness)
#         roi_corners = np.array([[(x1_left, y1_left), (x2_left, y2_left), (x2_right, y2_right), (x1_right, y1_right)]], dtype=np.int32)
#         cv2.fillPoly(mask, roi_corners, (255, 255, 255))
        
#         img = cv2.bitwise_and(image, mask)
        
    
#     return img

def hough_lines(image,img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    #cv2.imwrite('/home/eleensmathew/TrafficAnalysis/output_frame/ppp.png', img)
    #line_img=draw_lines(image,line_img, lines)
    line_img=draw_polygon(image,line_img, vertices)
    return line_img

def weighted_img(img, intial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(intial_img, α, img, β, λ)

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
    
        print(vertices)
        
        masked_edges = region_of_interest(edges,vertices)

        rho = 1
        theta = np.pi/180
        threshold = 20
        min_line_length = 40
        max_line_gap = 300


        lines = hough_lines(image,masked_edges, rho, theta, threshold,
                                min_line_length, max_line_gap)

        
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

    if len(contours) == 0:#no motion
        return False
    else:
        return True
def get_frames():
    image_files = os.listdir('/home/eleensmathew/TrafficAnalysis/output_frame/')
    image_files.sort()
    prev_frame = cv2.imread(os.path.join('/home/eleensmathew/TrafficAnalysis/output_frame/', image_files[0]))
    for image_file in image_files[1:]:
       
        curr_frame = cv2.imread(os.path.join('/home/eleensmathew/TrafficAnalysis/output_frame/', image_file))
        print(detect_motion(prev_frame, curr_frame))
        prev_frame = curr_frame
# Load the image

extract_frames('/home/eleensmathew/TrafficAnalysis/data/videos/recording (online-video-cutter.com).mp4', '/home/eleensmathew/TrafficAnalysis/data/frames/', frame_interval=60)
get_vertices('/home/eleensmathew/TrafficAnalysis/data/frames/frame0.jpg')
read()
get_frames()