import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

cap = cv.VideoCapture(args.image)
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

step = 16

vectors = list()    #추출된 시각(스탬프프),속도, 가속도, 각도(방향),아래로 이동했는지 여부(각도로 판별)
old_speed = 0

#시간 함수 사용으로 수정 (ms 단위)
dt = 1/cap.get(cv.CAP_PROP_FPS)

while True:
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    #가우시안 삭제필요
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2)
    
    #영상의 높이, 넓이
    h,w = frame2.shape[:2]
    
    #16픽셀로 나누어 움직임 계산(그리드로 벡터 시각화한 코드 응용)
    #step/2:h:step -> 8부터 영상의 높이까지 16의 간격
    #step/2:w:step -> 8부터 영상의 넓이까지 16의 간격
    idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.int64)
    
    #x축의 인덱스와 y축의 인덱스 값을 열 기준으로 결합 -> (x축 인덱스, y축 인덱스) 형태, 그 후 2차원 배열로 변환 
    indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)
    
    for x, y in indices:
        dx,dy = flow[y, x].astype(np.int64)
        #speed, acc 수정 & 벡터 추출 시간, 평면 좌표계 -> 속도 제공됨(직접 x)
        speed = float(np.sqrt(dx**2 + dy**2)/dt)
        acceleration = float((speed - old_speed)/dt)
        isDownwards = None
        angle = float(np.arctan2(dy,dx)*(180.0/np.pi))
        
        old_speed = speed
        """if dy > 0:
            isDownwards = True
        else:
            isDownwards = False"""
        vectors.append((speed,acceleration,isDownwards,angle))
  
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    prvs = next
cv.destroyAllWindows()

print(vectors)