import numpy as np
import cv2 as cv
import argparse
import time

## 쓰레드 분리 (async()?), 그래도 렉걸리면 mediapipe 제거, 그리드 제거 고려(goodFeaturesToTrack 사용 고려), (업피라미드, 다운 피라미드 동시사용 고려)
## vec -tiemstamp별 x,y,속도,각도,낙상여부
code_start = time.time()

"""import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils"""

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
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                  flags = 0,
                  minEigThreshold = 1e-4)

# Take first frame and find corners in it
ret, old_frame = cap.read()

old_t = time.time()

step = 16
h,w = old_frame.shape[:2]
idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.int64)
indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)

idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.float32)
p0 =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,1,2)

#old frame을 회색으로 변환 (밝기 향상성)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
#p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

#속도, 가속도, 아래로 이동했는지 여부
vectors = list()

#speed = list()    
old_speed = np.zeros((h, w), dtype=np.float32)

isStart = True

def calcVec(new,old):
    x, y = old.ravel()
    x = int(x)
    y = int(y)
    dx,dy = new.ravel() - old.ravel()    
    #vectors.append()
    return x,y,dx,dy

while True:
    ret, frame = cap.read()    
    
    if not ret:
        print('No frames grabbed!')
        break
    
    if isStart:
        start = time.time()
        #print(start - code_start)
        isStart = False
    
    new_t = time.time()
    
    dt = new_t - old_t
    
    if dt == 0:
        #print(new_t,old_t)
        dt += 1/60
        
        
    #frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    #results = pose.process(frame_rgb)
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    
    
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray,p0, None, **lk_params)
    
    mask = st==1#((st==1) & (err<8))
    
    # Select good points
    if p1 is not None:
        good_new = p1[mask]
        good_old = p0[mask]
    
    #temp = list()
    #flow = np.zeros((h,w,2),dtype=np.float32)
    
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new,good_old)):
        x,y,dx,dy = calcVec(new,old)
        cv.line(frame, (x,y), (x+int(dx), y+int(dy)), (0,0,255),2, cv.LINE_AA )
        """speed = float(np.sqrt(dx**2 + dy**2)/dt)
        acceleration = float((speed - old_speed[y,x])/dt)
        old_speed[y,x] = speed
        isDownwards = False
        angle = float(np.arctan2(dy,dx)*(180.0/np.pi))
        if angle < -15 and angle >-155:
                isDownwards = True"""
        #vectors.append((new_t,speed,acceleration,isDownwards,angle))
        
    for x, y in indices:
        cv.circle(frame, (x,y), 1, (0,255,0), -1)
        #dx,dy = flow[y, x].astype(np.int64)
        
    #cv.imshow('frame', frame)
    
    #esc키를 누르면 종료
    k = cv.waitKey(30) & 0xff
    
    if k == 27:
        break
    
    # Now update the previous frame and previous points
    old_t = new_t
    old_gray = frame_gray.copy()
    
cv.destroyAllWindows()
end = time.time()-start

#print(vectors)
print(end)