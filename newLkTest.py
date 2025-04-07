import numpy as np
import cv2 as cv
import argparse
import time
import mediapipe as mp

## 목표 그리드 형식으로 변경, 해상도별로 (원본제외 4단계까지) 계산 (피라미드),  
start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

termination = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
feature_params = dict(maxCorners = 200, qualityLevel = 0.01, minDistance = 7, blockSize = 7)
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = termination)


track_len =2
detect_interval = 5
tracks = []
cam = cv.VideoCapture(args.image)
frame_idx = 0
blackscreen = False
width = int(cam.get(3))
height = int(cam.get(4))

step = 16
idx_y,idx_x = np.mgrid[step/2:height:step,step/2:width:step].astype(np.int64)
indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)

old_t = 0
new_t = 0
vector = list()

while True:
    ret, frame = cam.read()
    
    if not ret:
        print("No frame grabbed")
        break
    
    if frame_idx==0:
        old_t = time.time()
    else:
        new_t = time.time()
    
    dt = new_t - old_t
    
    frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = pose.process(frame_rgb) 
            
    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    vis = frame.copy()
            
    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        #print(p0.shape)
        p1, st, err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r ,st, err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                
        d = abs(p0 - p0r).reshape(-1,2).max(-1)
        good = d < 1
                
        new_tracks = []
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1,2),good):
            if not good_flag:
                continue
                    
            tr.append((x,y))
            
            if len(tr) > track_len:
                #print(tr,len(tr),tr[0],"\n")
                del tr[0]
                    
            new_tracks.append(tr)
            #print(tr[1],tr[0])
            dx= tr[1][0] - tr[0][0]
            dy = tr[1][1] -tr[0][1]
            #print(dx,dy)
            speed = float(np.sqrt(dx**2 + dy**2)/dt)
            #acceleration = float((speed - old_speed[i])/dt)
            isDownwards = False
            angle = float(np.arctan2(dy,dx)*(180.0/np.pi))
            if angle>-165 and angle <-15:
                isDownwards = True
            #print(angle)
            vector.append((new_t,speed,isDownwards))
            cv.circle(vis,(int(x),int(y)), 2, (0,255,0), -1)
                    
        tracks = new_tracks
        #print(tr)
        cv.polylines(vis,[np.int32(tr) for tr in tracks], False, (0,255,0))
                
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x,y in [np.int32(tr[-1]) for tr in tracks]:
            cv.circle(mask,(x,y), 5, 0, -1)
        p = cv.goodFeaturesToTrack(frame_gray,mask=mask,**feature_params)
        if p is not None:
            for x,y in np.float32(p).reshape(-1,2):
                tracks.append([(x,y)])
    
    #if frame_idx == 0:
        #print(tracks)
    frame_idx += 1
    prev_gray = frame_gray
            
    cv.imshow("frame",vis)
            
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break 
        
cam.release()
cv.destroyAllWindows()


end = time.time()-start
print(vector)
print(end)
