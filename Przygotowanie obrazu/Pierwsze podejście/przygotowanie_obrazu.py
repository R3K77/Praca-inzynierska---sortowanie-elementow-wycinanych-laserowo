import cv2
import numpy as np

# Load images
image1 = cv2.imread('1.png')
image2 = cv2.imread('2.png')
image3 = cv2.imread('3.png')

image1 = cv2.resize(image1, (1500, int(image1.shape[0] * 1500 / image1.shape[1])))


width = image1.shape[1]
height = image1.shape[0]
MidX = width/2
MidY = height/2
maxW = width/2
maxH = height/2


dst = np.array([
    [0, 0],
    [maxW-1, 0],
    [maxW-1, maxH-1],
    [0, maxH-1]], dtype = "float32")


def get_canny_edge(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        # return the edged image
        return edged   
    
def correct_perspective(frame, pts):
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (width/2, height/2))
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(warped)
    threshold = (min_val + max_val) / 2
    ret, warped = cv2.threshold(warped, threshold, 255, cv2.THRESH_BINARY)
    return warped
    

# Check if images are loaded successfully
if image1 is None or image2 is None or image3 is None:
    print('Failed to load images.')
else:
    print('Images loaded successfully.')
    
    # Create a window to display the images
    cv2.namedWindow('Image Viewer')
    
    # Resize images
    image2 = cv2.resize(image2, (1500, int(image2.shape[0] * 1500 / image2.shape[1])))
    image3 = cv2.resize(image3, (1500, int(image3.shape[0] * 1500 / image3.shape[1])))
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    blurred1 = cv2.GaussianBlur(gray1,(3,3),0)
    
    image1 = get_canny_edge(blurred1, sigma=1)
    
    # Find contours in the image
    contours, _ = cv2.findContours(image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    
    

    # Draw the rectangle on the image
    cv2.rectangle(image1, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Draw four red dots in the corners of the rectangle
    cv2.circle(blurred1, (x, y), 5, (0, 0, 255), -1)
    cv2.circle(blurred1, (x+w, y), 5, (0, 0, 255), -1)
    cv2.circle(blurred1, (x, y+h), 5, (0, 0, 255), -1)
    cv2.circle(blurred1, (x+w, y+h), 5, (0, 0, 255), -1)


    # Define a function to handle trackbar events
    def on_trackbar(pos):
        if pos == 0:
            cv2.imshow('Image Viewer', blurred1)
        elif pos == 1:
            cv2.imshow('Image Viewer', image1)
        elif pos == 2:
            cv2.imshow('Image Viewer', gray1)

    # Create a trackbar
    cv2.createTrackbar('Image', 'Image Viewer', 0, 2, on_trackbar)

    # Show the initial image
    cv2.imshow('Image Viewer', blurred1)

    # Wait for a key press to exit
    cv2.waitKey(0)

    # Destroy the window
    cv2.destroyAllWindows()
    
    