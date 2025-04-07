import numpy as np
import mediapipe as mp
import cv2 as cv
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

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

#old frame을 회색으로 변환 (밝기 향상성)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

print(p0.shape)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

#속도, 가속도, 아래로 이동했는지 여부
vectors_h = list()
vectors_w = list()

#speed = list()    
old_speed = list()
old_speed.append(0)

p0_head = None
p0_waist = None

while True:
    ret, frame = cap.read()
    
    new_t = time.time()
    
    dt = new_t - old_t
    
    if not ret:
        print('No frames grabbed!')
        break
    frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS)
    
    if results.pose_landmarks is not None:
        if p0_head is None and p0_waist is None:
            p0_head = np.array([[l.x, l.y] for l in results.pose_landmarks.landmark[0:11]],np.float32).reshape(-1,1,2)
            #print(p0_head.shape)
            p0_waist = np.array([[l.x, l.y] for l in results.pose_landmarks.landmark[23:25]],np.float32).reshape(-1,1,2)
            continue

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    
        
        #print(p0.shape,p1.shape)
        # calculate optical flow
        p1_h, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_head, None, **lk_params,flags=0)
        #print(p1_h.shape)
        p1_w, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_waist, None, **lk_params,flags=0)
        #p0r, st, err = cv.calcOpticalFlowPyrLK(frame_gray,old_gray,p1,None,**lk_params,flags=0)
        
        #print("p0 : ",p0,"p0r : ",p0r,"d : ",p0 - p0r)
        # Select good points
        """if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]"""
        
        #print(good_new)
        #temp = list()
        for (new,old) in zip(p1_h,p0_head):
            a, b = new.ravel()
            c, d = old.ravel()
            #mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 1, (0,255, 0), -1)
            cv.line(frame, (int(a), int(b)), (int(c), int(d)), (0,255, 0), 2,cv.LINE_AA)
            
            #frame에서의 point와 old frame의 point의 차 = dx, dy
            #print(new,old)
            dx, dy = new.ravel() - old.ravel()
            speed = float(np.sqrt(dx**2 + dy**2)/dt)
            #acceleration = float((speed - old_speed[i])/dt)
            isDownwards = False
            angle = float(np.arctan2(dy,dx)*(180.0/np.pi))
            
            #speed.append(speed)
            old_speed.append(speed)
            #dx 양수: 오른쪽 이동, dy 증가: 아래로 이동
            #angle 음수(ex) -15 ~155 )
            if angle < -15 and angle >-155:
                isDownwards = True
                if speed > 17:   
                    print(new_t)
            
            #temp.append([(a,b)])
            vectors_h.append((new_t,speed,isDownwards,angle))
        for (new,old) in zip(p1_w,p0_waist):
            a, b = new.ravel()
            c, d = old.ravel()
            #mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 2, (0,0,255), -1)
            cv.line(frame, (int(a), int(b)), (int(c), int(d)), (0,0,255), 2)
            
            #frame에서의 point와 old frame의 point의 차 = dx, dy
            #print(new,old)
            dx, dy = new.ravel() - old.ravel()
            speed = float(np.sqrt(dx**2 + dy**2)/dt)
            #acceleration = float((speed - old_speed[i])/dt)
            isDownwards = False
            angle = float(np.arctan2(dy,dx)*(180.0/np.pi))
            
            #speed.append(speed)
            old_speed.append(speed)
            #dx 양수: 오른쪽 이동, dy 증가: 아래로 이동
            #angle 음수(ex) -15 ~155 )
            if angle < -15 and angle >-155:
                isDownwards = True
            
            #temp.append([(a,b)])
            vectors_w.append((new_t,speed,isDownwards,angle))
            
        p0_head = p1_h
        p0_waist = p1_w
    # draw the tracks
    """for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        #mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
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
        
        temp.append([(a,b)])
        vectors.append((speed,acceleration,isDownwards,angle))"""  
        
    #img = cv.add(frame, mask)        
    
    #print([np.int32(i) for i in temp])
    #cv.polylines(img,[np.int32(i) for i in temp],False,color[i].tolist())      
    
    cv.imshow('frame', frame)
    
    #p0_head = p1_h
    #p0_waist = p1_w
    
    #esc키를 누르면 종료
    k = cv.waitKey(30) & 0xff
    
    if k == 27:
        break
    
    # Now update the previous frame and previous points
    old_t = new_t
    #old_gray = frame_gray.copy()
    #p0 = np.copy(p1)
    
cv.destroyAllWindows()

#
#print(vectors_h)
print(max(v[1] for v in vectors_h if v[2] == True))
#print(vectors_w)