import numpy as np
import cv2 as cv
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

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

old_t = time.time()

step = 16
h,w = old_frame.shape[:2]
idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.float32)
indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,1,2)

#old frame을 회색으로 변환 (밝기 향상성)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

print(p0.dtype)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

#속도, 가속도, 아래로 이동했는지 여부
vectors = list()

#speed = list()    
old_speed = list()
old_speed.append(0)

while True:
    ret, frame = cap.read()
    
    new_t = time.time()
    dt = new_t - old_t
    if dt == 0:
        print(new_t,old_t)
    if not ret:
        print('No frames grabbed!')
        break
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    
    
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray,indices, None, **lk_params,flags=0)
    
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
    
    #print(p1)
    temp = list()
    # draw the tracks
    """for (new, old) in enumerate(zip(good_new, p0)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 2, color[i].tolist(), -1)

        
        #frame에서의 point와 old frame의 point의 차 = dx, dy
        #print(new,old)
        dx, dy = new.ravel() - old.ravel()
        speed = float(np.sqrt(dx**2 + dy**2)/dt)
        acceleration = float((speed - old_speed[i])/dt)
        isDownwards = False
        angle = float(np.arctan2(dy,dx)*(180.0/np.pi))
        
        #speed.append(speed)
        old_speed.append(speed)
        #dx 양수: 오른쪽 이동, dy 증가: 아래로 이동
        #angle 음수(ex) -15 ~155 )
        if angle < -15 and angle >-155:
            isDownwards = True
        
        temp.append(i)
        vectors.append((speed,acceleration,isDownwards,angle))  
        
    img = cv.add(frame, mask)        
    
    print(temp)
    cv.polylines(img,[np.int32(i) for i in temp],False,color[i].tolist())      
    
    cv.imshow('frame', img)
    
    #esc키를 누르면 종료
    k = cv.waitKey(30) & 0xff
    
    if k == 27:
        break
    
    # Now update the previous frame and previous points
    old_t = new_t
    old_gray = frame_gray.copy()
    p0 = np.copy(p1)"""
    
cv.destroyAllWindows()

#print(vectors)