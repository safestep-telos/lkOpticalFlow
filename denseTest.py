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

vectors = list()    #속도, 가속도, 아래로 이동했는지 여부
old_speed = 0
dt = 1/cap.get(cv.CAP_PROP_FPS)

while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    h,w = frame2.shape[:2]
    
    idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.float64)
    indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)
    
    for dx, dy in indices:
        speed = np.sqrt(dx**2 + dy**2)
        acceleration = (speed - old_speed)/dt
        isDownwards = False
        if dy > 0:
            isDownwards = True
        vectors.append((speed,acceleration,isDownwards))
  
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    """elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)"""
    prvs = next
cv.destroyAllWindows()

print(vectors)