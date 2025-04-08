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
                  maxLevel = 5,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                  flags = 0,
                  minEigThreshold = 1e-4)

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()

old_t = time.time()

step = 16
h,w = old_frame.shape[:2]
idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.int64)
indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)

idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.float32)
p0 =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)

#old frame을 회색으로 변환 (밝기 향상성)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
#p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

#속도, 가속도, 아래로 이동했는지 여부
vectors = list()

#speed = list()    
old_speed = list()
old_speed.append(0)

"""flow = []
flow = np.array(flow,np.float32)"""

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
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray,p0, None, **lk_params)
    
    #print(p1)
    # Select good points
    if p1 is not None:
        #print(p1.shape,st.shape)
        good_new = p1.reshape(-1,1,2)[st==1]
        good_old = p0.reshape(-1,1,2)[st==1]
    
    temp = list()
    flow = np.zeros((h,w,2),dtype=np.float32)
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new,good_old)):
        x,y = old.ravel()
        x = int(x)
        y = int(y)
        flow[y,x] = new.ravel() - old.ravel()
        dx, dy = new.ravel() - old.ravel()
        dx = int(dx)
        dy = int(dy)
        #frame에서의 point와 old frame의 point의 차 = dx, dy
        
    for x, y in indices:
        cv.circle(frame, (x,y), 1, (0,255,0), -1)
        dx,dy = flow[y, x].astype(np.int64)
        cv.line(frame, (x,y), (x+dx, y+dy), (0,255, 0),2, cv.LINE_AA )
        #cv.line(frame, (x,y), (x+dx, y+dy), (0,255, 0),2, cv.LINE_AA )
        
    #print(flow)   
     
    #img = cv.add(frame, mask)        
    
    #print(temp)
    #cv.polylines(img,[np.int32(i) for i in temp],False,(0,255,0))      
    
    cv.imshow('frame', frame)
    
    #esc키를 누르면 종료
    k = cv.waitKey(30) & 0xff
    
    if k == 27:
        break
    
    # Now update the previous frame and previous points
    #p0 = good_new.reshape(-1,1,2)
    #old_t = new_t
    #old_gray = frame_gray.copy()
    
cv.destroyAllWindows()

#print(vectors)