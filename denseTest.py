import numpy as np
import cv2 as cv
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

cap = cv.VideoCapture(args.image)
ret, frame1 = cap.read()

old_t = time.time()

prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

step = 16

vectors = list()    #속도, 가속도, 아래로 이동했는지 여부
old_speed = 0
#dt = 1/cap.get(cv.CAP_PROP_FPS)

while True:
    ret, frame2 = cap.read()
    
    new_t = time.time()
    dt = new_t - old_t
    if not ret:
        print('No frames grabbed!')
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    #print(flow)
    #영상의 높이, 넓이
    h,w = frame2.shape[:2]
    
    #16픽셀로 나누어 움직임 계산(그리드로 벡터 시각화한 코드 응용)
    #step/2:h:step -> 8부터 영상의 높이까지 16의 간격
    #step/2:w:step -> 8부터 영상의 넓이까지 16의 간격
    idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.int64)
    
    #x축의 인덱스와 y축의 인덱스 값을 열 기준으로 결합 -> (x축 인덱스, y축 인덱스) 형태, 그 후 2차원 배열로 변환 
    indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)
  
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1],angleInDegrees=False)
    
    for x, y in indices:
        #dx,dy = flow[y, x].astype(np.int64)
        #print(mag[y,x])
        speed = float(mag[y,x])
        angle = float(ang[y,x])
        acceleration = float((speed - old_speed)/dt)
        isDownwards = None
        #angle = float(np.arctan2(dy,dx)*(180.0/np.pi))
        """if dy > 0:
            isDownwards = True"""
        if angle > np.pi and angle < 2*np.pi:
            print(angle)
        vectors.append((speed,angle))
    #print("mag: ",mag,"ang: ",ang)
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    prvs = next
cv.destroyAllWindows()

#print(vectors)