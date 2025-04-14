import numpy as np
import cv2 as cv
import argparse
import time


## 쓰레드 분리 (async()?), 그래도 렉걸리면 mediapipe 제거, 그리드 제거 고려(goodFeaturesToTrack 사용 고려), (업피라미드, 다운 피라미드 동시사용 고려)
## vec -tiemstamp별 x,y,속도,각도,낙상여부
code_start = time.time()

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

def calcVec(new,old):
    x, y = old.ravel()
    x = int(x)
    y = int(y)
    dx,dy = new.ravel() - old.ravel()    
    #vectors.append()
    return x,y,dx,dy

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
    
    mask = st==1#((st==1) & (err<8))
    
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
        #vectors.append((new_t,speed,acceleration,isDownwards,angle))
        
        #dx,dy = flow[y, x].astype(np.int64)
    end_time = time.time()
    total_processing_time += (end_time - start_time)
    frame_count += 1
    cv.imshow('frame', frame)
    
    #esc키를 누르면 종료
    k = cv.waitKey(30) & 0xff
    
    if k == 27:
        break
    
    # Now update the previous frame and previous points
    #old_t = new_t
    old_gray = frame_gray
    #p0 = good_new.reshape(-1,1,2)
    
cv.destroyAllWindows()

print(total_processing_time)
#print(vectors)