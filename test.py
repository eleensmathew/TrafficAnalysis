import cv2
import numpy as np

# Global variables


drawing = False # true if mouse is pressed
ix, iy = -1, -1
vertices = []
def get_vertices(img_path):
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
        

    # Load the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (960, 540))

    # Create a black image, a window and bind the function to window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_roi)

    while(1):
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to exit
            break

    cv2.destroyAllWindows()

    # Convert the vertices to a numpy array
    vertices = np.array([vertices], dtype=np.int32)
    return vertices
get_vertices('/home/eleensmathew/TrafficAnalysis/data/solidWhiteCurve (1).jpg')
print(vertices)
