import numpy as np
import cv2 as cv
import argparse
import time

"""import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()"""

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

cap = cv.VideoCapture(args.image)

print(cap.get(cv.CAP_PROP_FRAME_COUNT)/cap.get(cv.CAP_PROP_FPS))

# params for ShiTomasi corner detection
"""feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )"""

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 5,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame = cap.read()
print(cap.get(cv.CAP_PROP_FRAME_COUNT)/cap.get(cv.CAP_PROP_FPS))
old_t = time.time()

step = 16
h,w = old_frame.shape[:2]
idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.int64)
indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)

idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.float32)
p0 =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,1,2)

#old frame을 회색으로 변환 (밝기 향상성)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

#속도, 가속도, 아래로 이동했는지 여부
vectors = list()

#speed = list()    
old_speed = np.zeros((h, w), dtype=np.float32)

frame_count = 0
total_processing_time = 0

while True:
    ret, frame = cap.read()    
    start_time = time.time()
    
    if not ret:
        print('No frames grabbed!')
        break
    
    new_t = time.time()
    
    dt = new_t - old_t
    
    if dt == 0:
        dt += 1/60        
        
    #frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    #results = pose.process(frame_rgb)
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    
    
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray,p0, None, **lk_params)
    
    mask = ((st==1) & (err<8))
    
    # Select good points
    if p1 is not None:
        good_new = p1[mask]
        good_old = p0[mask]
    
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new,good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        dx, dy = new.ravel() - old.ravel()
        speed = float(np.sqrt(dx**2 + dy**2))
        #acceleration = float((speed - old_speed[y,x])/dt)
        #old_speed[y,x] = speed
        isDownwards = False
        angle = float(np.arctan2(dy,dx)*(180.0/np.pi))
        if angle < -15 and angle >-155:
                isDownwards = True
        if speed > 1 and speed < 20:
            cv.line(frame, (int(c), int(d)), (int(a), int(b)), (0,0,255),2, cv.LINE_AA )
        
    """for x, y in indices:
        cv.circle(frame, (x,y), 2, (0,255,0), -1)"""
        
    flow = cv.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    """for x, y in indices:
        #cv.circle(frame_gray, (x,y), 2, (0,255,0), -1)
        dx,dy = flow[y, x].astype(np.int64)
        cv.line(frame, (x,y), (x+dx, y+dy), (0,255,0),2, cv.LINE_AA )"""
        
    end_time = time.time()
    total_processing_time += (end_time - start_time)
    frame_count += 1
    
    cv.imshow('frame', frame)
    
    #esc키를 누르면 종료
    k = cv.waitKey(1) & 0xff
    
    if k == 27:
        break
    
    old_gray = frame_gray
    
cv.destroyAllWindows()

print(total_processing_time)