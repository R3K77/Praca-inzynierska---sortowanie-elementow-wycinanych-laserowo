import numpy as np
import cv2
from skimage import data, filters

# ------------------ Skrypt do wykrywania nowych elementów na obrazie - TEST ------------------- #

#na podstawie
#https://learnopencv.com/simple-background-estimation-in-videos-using-opencv-c-python/
#https://learnopencv.com/moving-object-detection-with-opencv/

# Open Video
cap = cv2.VideoCapture('../ZdjeciaElementy/bgr_video.mp4')

# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

# Display median frame
cv2.imshow('frame', medianFrame)
cv2.waitKey(0)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

#Do zapisania wideo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Loop over all frames
ret = True
out = None
while (ret):
    # Read frame
    ret, frame = cap.read()
    # Convert current frame to grayscale
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate absolute difference of current frame and
        # the median frame
        dframe = cv2.absdiff(frame, grayMedianFrame)
        # Treshold to binarize
        th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
        # set the kernal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Apply erosion
        mask_eroded = cv2.morphologyEx(dframe, cv2.MORPH_OPEN, kernel)
        # mask_eroded = dframe
        contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 200  # Define your minimum area threshold
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        frame_out = frame.copy()
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            frame_out = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 200), 3)

        if dframe.shape[0] != frame.shape[0]:
            height = min(dframe.shape[0], frame.shape[0])
            dframe = cv2.resize(dframe, (dframe.shape[1], height))
            frame = cv2.resize(frame, (frame.shape[1], height))

        combined_image = cv2.vconcat([mask_eroded, frame_out])
        combined_image = cv2.resize(combined_image, (combined_image.shape[1] // 2, combined_image.shape[0] // 2))

        if out is None:
            out = cv2.VideoWriter('../../output.mp4', fourcc, 20.0, (combined_image.shape[1], combined_image.shape[0]))

        out.write(combined_image)
        cv2.imshow('Combined Image', combined_image)
        cv2.waitKey(20)
    else:
        break

# Release video object
cap.release()
out.release()
# Destroy all windows
cv2.destroyAllWindows()