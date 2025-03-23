import cv2 as cv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

body_cascade = cv.CascadeClassifier('haarcascade_fullbody.xml')

#optical flow 부분
cap = cv.VideoCapture(args.image)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print('frame grab fail')
        break
    
    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    blur = cv.GaussianBlur(frame_gray, (5, 5), 0)

    body = body_cascade.detectMultiScale(blur, 1.05, 2, 0, (15, 15))

    for (x,y,w,h) in body:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        print('x : ',x,',y : ',y,', w : ',w,', h : ',h)
        
    cv.imshow('frame',frame)
    if cv.waitKey(5) & 0xFF == 27:
        break
cv.waitKey()
cv.destroyAllWindows()