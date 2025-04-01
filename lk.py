import numpy as np
import cv2 as cv
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

termination = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
feature_params = dict(maxCorners = 200, qualityLevel = 0.01, minDistance = 7, blockSize = 7)
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = termination)

class App:
    def __init__(self,video_src):
        self.track_len =10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv.VideoCapture(video_src)
        self.frame_idx = 0
        self.blackscreen = False
        self.width = int(self.cam.get(3))
        self.height = int(self.cam.get(4))
        
    def run(self):
        while True:
            ret, frame = self.cam.read()
            
            if not ret:
                print("No frame grabbed")
                break
            
            frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            vis = frame.copy()
            
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                print(p0.shape)
                p1, st, err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r ,st, err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                
                d = abs(p0 - p0r).reshape(-1,2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1,2),good):
                    print(tr,(x,y))
                    if not good_flag:
                        continue
                    
                    tr.append((x,y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    
                    new_tracks.append(tr)
                    cv.circle(vis,(int(x),int(y)), 2, (0,255,0), -1)
                    
                self.tracks = new_tracks
                #print(self.tracks)
                cv.polylines(vis,[np.int32(tr) for tr in self.tracks], False, (0,255,0))
                
            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x,y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask,(x,y), 5, 0, -1)
                p = cv.goodFeaturesToTrack(frame_gray,mask=mask,**feature_params)
                if p is not None:
                    for x,y in np.float32(p).reshape(-1,2):
                        self.tracks.append([(x,y)])
            
            self.frame_idx += 1
            self.prev_gray = frame_gray
            
            cv.imshow("frame",vis)
            
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break 
        
        self.cam.release()

video_src = args.image
App(video_src).run()
cv.destroyAllWindows()