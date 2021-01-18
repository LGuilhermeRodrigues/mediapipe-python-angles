import sys
import mediapipe as mp
import cv2
from utils import *

reps_counting = False
reps = {LEFT_ELBOW:0, LEFT_SHOULDER:0}

def open_cv_loop(video_capture,loop_function):
    cap = cv2.VideoCapture(video_capture)
    while(cap.isOpened()):
        ret, video_frame = cap.read()
        new_h = 400
        new_w = int(new_h*video_frame.shape[1]/video_frame.shape[0])
        video_frame = cv2.resize(video_frame, (new_w, new_h))
        video_frame = loop_function(cv2.flip(video_frame, 1))
        cv2.imshow('video_frame',video_frame)
        key = cv2.waitKey(1)
        if  key != -1 or not cv2.getWindowProperty('video_frame', cv2.WND_PROP_VISIBLE):
            if key & 0xFF == ord('t'):
                global reps_counting
                global reps
                reps = {LEFT_ELBOW:0, LEFT_SHOULDER:0}
                reps_counting = True
            else:
                break
    cap.release()
    cv2.destroyAllWindows()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
def process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    return results
def draw_pose(video_frame, results=None):
    if not results:
        results = process(video_frame)
    mp_drawing.draw_landmarks(video_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return video_frame

def run_video():
    open_cv_loop('youtube.mp4',draw_pose)

def run_camera():
    open_cv_loop(1,draw_pose)

last_angle = {LEFT_ELBOW:0, LEFT_SHOULDER:0}
def repetitions(image):
    global reps
    global last_angle
    results = process(image)
    if not results.pose_landmarks:
        return image
    image = draw_pose(image, results)
    left_wrist = results.pose_landmarks.landmark[LEFT_WRIST]
    left_shoulder = results.pose_landmarks.landmark[LEFT_SHOULDER]
    left_elbow = results.pose_landmarks.landmark[LEFT_ELBOW]
    angle_2d = angle(left_shoulder,left_elbow,left_wrist)
    angle_3d_ = angle_3d(left_shoulder,left_elbow,left_wrist)
    angle_sholder = angle(results.pose_landmarks.landmark[LEFT_HIP],left_shoulder,left_elbow)
    print(f'2D: {angle_2d:.2f}   3D: {angle_3d_:.2f}')
    if reps_counting:
        angle_change = 30
        if last_angle[LEFT_ELBOW] > angle_change and angle_2d < angle_change:
            reps[LEFT_ELBOW]+=1
        cv2.putText(image,f'Elbow: {reps[LEFT_ELBOW]}',(300,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
        angle_change = 80
        if last_angle[LEFT_SHOULDER] < angle_change and angle_sholder > angle_change:
            reps[LEFT_SHOULDER]+=1
        cv2.putText(image,f'Sholder: {reps[LEFT_SHOULDER]}',(300,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    last_angle[LEFT_ELBOW] = angle_2d
    last_angle[LEFT_SHOULDER] = angle_sholder
    return image

def run_repetitions():
    open_cv_loop(1,repetitions)


if __name__ == "__main__":
    if len(sys.argv[1:]):
        run_repetitions()
    else:
        run_video()
