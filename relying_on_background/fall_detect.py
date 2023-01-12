import cv2
import time
# Reference repository here - https://github.com/EikeSan/video-fall-detection
# DRAWBACKS OF THIS APPROACH: 
# 1. CANNOT BE APPLIED WHERE THE BACKGROUND IS MOVING A LOT. 
# 2. NOT DETECTING PERSONS. It detects anything that changes it's position instantly. 

# 0 for web camera
cap = cv2.VideoCapture(0)
# time.sleep(2)
# to remove background https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html
foreground_background = cv2.createBackgroundSubtractorMOG2()
j = 0

while True:
    # reading the video frame by frame
    ret, frame = cap.read()
    
    # converting each frame into gray scale and subtracting the background
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = foreground_background.apply(gray_frame)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    if contours:
        areas = []
        for contour in contours:
            ar = cv2.contourArea(contour)
            areas.append(ar)
        max_area = max(areas, default = 0)
        max_area_index = areas.index(max_area)
        cnt = contours[max_area_index]

        M = cv2.moments(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(fg_mask, [cnt], 0, (255, 255, 0), 3, maxLevel = 0)
        

        if h < w:
            j += 1
        if j > 10:
            cv2.putText(frame, 'FALL', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if h > w:
            j = 0
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    # to end the live video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.waitKey(1)
cv2.destroyAllWindows()