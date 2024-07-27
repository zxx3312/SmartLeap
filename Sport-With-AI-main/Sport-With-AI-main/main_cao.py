import os

import cv2
import numpy as np
import mediapipe as mp
from body_part_angle import BodyPartAngle
from types_of_exercise import TypeOfExercise

# 设置视频地址
video_source = "videos/long-jump.mp4"

# 设置动作类型（输出文件的名称）
exercise_type = "cjd_test1"

# 初始化 Mediapipe 相关
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 读取视频文件
cap = cv2.VideoCapture(video_source)
cap.set(3, 800)  # 设置宽度
cap.set(4, 480)  # 设置高度

# 存储所有帧的关键点数据
all_landmarks = []

# 设置 Mediapipe Pose 模块
with mp_pose.Pose(min_detection_confidence=0.5,
                   min_tracking_confidence=0.5) as pose:
    counter = 0  # 运动计数
    status = True  # 运动状态
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = pose.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                # 存储当前帧的关键点数据
                frame_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks]
                all_landmarks.append(frame_landmarks)

                counter, status = TypeOfExercise(landmarks).calculate_exercise(
                    exercise_type, counter, status)
            except:
                pass

            # 渲染关键点
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(174, 139, 45), thickness=2, circle_radius=2),
            )

            cv2.imshow('Video', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("counter: " + str(counter))
                break
        except Exception as e:
            print(f"Error: {str(e)}")
            print("err counter: " + str(counter))
            break

    cap.release()
    cv2.destroyAllWindows()

# 转换为Numpy数组
all_landmarks_np = np.array(all_landmarks)
print(all_landmarks_np)

# 设置保存路径
output_dir = "../../Landmarks"
output_path = os.path.join(output_dir, "all_landmarks.npy")

# 确保目标文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 保存到指定文件
np.save(output_path, all_landmarks_np)

print(f"关键点数据保存至 {output_path}，共 {len(all_landmarks)} 帧")
