import mediapipe as mp
import cv2 as cv 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


cap = cv.VideoCapture(args.image)

while True:
    ret, frame = cap.read()
    if not ret:
        print('frame grab fail')
        break
    
    frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = pose.process(frame)   
    
    # 포즈 주석을 이미지 위에 그립니다.
    frame.flags.writeable = True
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    """mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS)"""
    print(results.pose_landmarks)
    cv.imshow('MediaPipe Pose', frame)
    
    if cv.waitKey(5) & 0xFF == 27:
        break
cv.waitKey()
cv.destroyAllWindows()
