import numpy as np
import cv2 as cv
import argparse

from collections import deque

def getVector():
    

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

cap = cv.VideoCapture(args.image)

body_cascade = cv.CascadeClassifier('haarcascade_fullbody.xml')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 3,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = None
#p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

vectors = list()    
old_speed = 0

#시간 변화 (영상의 경우: 영상의 fps 정보 기반, 실시간의 경우: time 함수 응용)
dt = 1/cap.get(cv.CAP_PROP_FPS)

xpos,ypos,width,height = deque(maxlen=1)

isFirst = False 
    
while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    
    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    blur = cv.GaussianBlur(frame_gray, (5, 5), 0)

    body = body_cascade.detectMultiScale(blur, 1.05, 2, 0, (15, 15))
    
    old_gray_roi = None
    frame_gray_roi = None
    
    if isFirst == True:
        if len(body) > 0:
            for (x,y,w,h) in body:
                old_gray_roi = old_gray[y:y+h, x:x+w]
                frame_gray_roi = frame_gray[y:y+h, x:x+w]
                xpos.append(x)
                ypos.append(y)
                width.append(w)
                height.append(h)
        else:
            old_gray_roi = old_gray
            frame_gray_roi = frame_gray
    else:
        if len(body) > 0:
            for (x,y,w,h) in body:
                old_gray_roi = old_gray[ypos[0]:ypos[0]+height[0], xpos[0]:xpos[0]+width[0]]
                frame_gray_roi = frame_gray[y:y+h,x:x+w]
                xpos.append(x)
                ypos.append(y)
                width.append(w)
                height.append(h)
        else:
            old_gray_roi = old_gray[ypos[0]:ypos[0]+height[0], xpos[0]:xpos[0]+width[0]]
            frame_gray_roi = frame_gray[ypos[0]:ypos[0]+height[0], xpos[0]:xpos[0]+width[0]]
    
    p0 = cv.goodFeaturesToTrack(old_gray_roi, mask = None, **feature_params)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray_roi, frame_gray_roi, p0, None, **lk_params)
    
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    
    for i, (new, old) in enumerate(zip(good_new,good_old)):
        #frame에서의 point와 old frame의 point의 차 = dx, dy
        dx, dy = new.ravel() - old.ravel()
        speed = float(np.sqrt(dx**2 + dy**2))
        acceleration = float((speed - old_speed)/dt)
        isDownwards = None
        angle = float(np.arctan2(dy,dx)*(180.0/np.pi))
        
        #dx 양수: 오른쪽 이동, dy 증가: 아래로 이동
        if dy > 0:
            isDownwards = True
            
        vectors.append((speed,acceleration,isDownwards,angle))
        
    img = cv.add(frame, mask)
    cv.imshow('frame', img)  
       
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    
cv.destroyAllWindows()