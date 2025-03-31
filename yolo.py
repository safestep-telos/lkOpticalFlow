import numpy as np
import cv2 as cv
import argparse

from collections import deque

def consturct_yolo():
    f = open('coco_names.txt','r')
    class_names= [line.strip() for line in f.readlines()]
    
    model = cv.dnn.readNet('yolov3.weights','yolov3.cfg')
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i-1] for i in model.getUnconnectedOutLayers()]
    
    return model, out_layers, class_names

def yolo_detect(frame, yolo_model,out_layers):
    height, width = frame.shape[0], frame.shape[1]
    test_img = cv.dnn.blobFromImage(frame,1.0/256,(448,448),(0,0,0),swapRB=True)
    
    yolo_model.setInput(test_img)
    output3 = yolo_model.forward(out_layers)
    
    box, conf, id = [],[],[]
    for output in output3:
        for vec85 in output:
            scores = vec85[5:]
            class_id = np.argmax(scores)
            confidence =   scores[class_id]
            if confidence > 0.5:
                certex, centry = int(vec85[0]*width), int(vec85[1]*height)
                w,h = int(vec85[2]*width), int(vec85[3]*height)
                x,y = int(certex-w/2), int(centry-h/2)
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)
                
    ind = cv.dnn.NMSBoxes(box,conf,0.5,0.4)
    
    objects = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]
    
    return objects


model, out_layers,class_names = consturct_yolo

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv.VideoCapture(args.image)

xpos,ypos,width,height = deque(maxlen=1)
isFirst = True

while True:
    ret, frame = cap.read()
    
    if not ret:
        print('No frames grabbed!')
        break
    
    res = yolo_detect(frame,model,out_layers)
    
    is_detected = False            
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    old_gray_roi = None
    frame_gray_roi = None
    
    if isFirst == True:
        for i in range(len(res)):
            x1,x2,y1,y2,confidence,id = res[i]
            if class_names[id] == 'person':
                is_detected = True
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                if w<0 or h<0:
                    print('!!')
        if is_detected == True:
            for (x,y,w,h) in body:
                old_gray_roi = old_gray[y:y+h, x:x+w]
                frame_gray[y:y+h, x:x+w]
                xpos.append(x)
                ypos.append(y)
                width.append(w)
                height.append(h)
        else:
            old_gray_roi = old_gray
            frame_gray_roi = frame_gray
    else:
        if len(body) > 0:
            for (x,y,w,h) in body:
                old_gray_roi = old_gray[ypos[0]:ypos[0]+height[0], xpos[0]:xpos[0]+width[0]]
                frame_gray_roi = frame_gray[y:y+h,x:x+w]
                xpos.append(x)
                ypos.append(y)
                width.append(w)
                height.append(h)
        else:
            old_gray_roi = old_gray[ypos[0]:ypos[0]+height[0], xpos[0]:xpos[0]+width[0]]
            frame_gray_roi = frame_gray[ypos[0]:ypos[0]+height[0], xpos[0]:xpos[0]+width[0]]
    
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    
    # draw the tracks
    """for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    
    img = cv.add(frame, mask)"""
    
    for i, (new, old) in enumerate(zip(good_new,good_old)):
        #frame에서의 point와 old frame의 point의 차 = dx, dy
        dx, dy = new.ravel() - old.ravel()
        speed = float(np.sqrt(dx**2 + dy**2))
        acceleration = float((speed - old_speed)/dt)
        isDownwards = None
        angle = float(np.arctan2(dy,dx)*(180.0/np.pi))
        
        #dx 양수: 오른쪽 이동, dy 증가: 아래로 이동
        if dy > 0:
            isDownwards = True
            
        vectors.append((speed,acceleration,isDownwards,angle))
        
    
    cv.imshow('frame', frame)
    
    #esc키를 누르면 종료료
    k = cv.waitKey(30) & 0xff
    
    if k == 27:
        break
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)