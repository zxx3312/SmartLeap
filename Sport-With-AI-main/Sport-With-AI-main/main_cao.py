## import packages
import cv2
import argparse
from utils import *
import mediapipe as mp
from body_part_angle import BodyPartAngle
from types_of_exercise import TypeOfExercise
import os

## 设置视频地址
## 请确保视频文件放在“videos”文件夹里！！！
current_dir = os.getcwd()
relative_path = "vieos/your_video_path"
video_source = os.path.join(current_dir, relative_path)

## 设置动作类型（输出文件的名称）
exercise_type = "cjd_test1"

## drawing body
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

## 读取视频文件
cap = cv2.VideoCapture(video_source)

cap.set(3, 800)  # width
cap.set(4, 480)  # height

## setup mediapipe
with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    counter = 0  # movement of exercise
    status = True  # state of move
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            # result_screen = np.zeros((250, 400, 3), np.uint8)

            frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
            ## recolor frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            ## make detection
            results = pose.process(frame)
            ## recolor back to BGR
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                counter, status = TypeOfExercise(landmarks).calculate_exercise(
                    exercise_type, counter, status)
            except:
                pass

            score_table(exercise_type, counter, status)

            ## render detections (for landmarks)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255),
                                       thickness=2,
                                       circle_radius=2),
                mp_drawing.DrawingSpec(color=(174, 139, 45),
                                       thickness=2,
                                       circle_radius=2),
            )

            cv2.imshow('Video', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("counter: " + str(counter))
                break
        except:
            print("err counter: " + str(counter))
            break

    cap.release()
    cv2.destroyAllWindows()
