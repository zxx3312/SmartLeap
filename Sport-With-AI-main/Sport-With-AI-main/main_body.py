import os

import cv2
import numpy as np
import mediapipe as mp
from body_part_angle import BodyPartAngle
from types_of_exercise import TypeOfExercise

# 设置视频地址
video_source = "../../videos/1.mp4"

# 设置动作类型（输出文件的名称）
exercise_type = "cjd_test1"

# 初始化 Mediapipe 相关
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 读取视频文件
cap = cv2.VideoCapture(video_source)
cap.set(3, 800)  # 设置宽度
cap.set(4, 480)  # 设置高度

# 关键点名称和对应索引
keypoint_names = {
    0: 'NOSE', 1: 'LEFT_EYE_INNER', 2: 'LEFT_EYE', 3: 'LEFT_EYE_OUTER',
    4: 'RIGHT_EYE_INNER', 5: 'RIGHT_EYE', 6: 'RIGHT_EYE_OUTER',
    7: 'LEFT_EAR', 8: 'RIGHT_EAR', 9: 'MOUTH_LEFT', 10: 'MOUTH_RIGHT',
    11: 'LEFT_SHOULDER', 12: 'RIGHT_SHOULDER', 13: 'LEFT_ELBOW', 14: 'RIGHT_ELBOW',
    15: 'LEFT_WRIST', 16: 'RIGHT_WRIST', 17: 'LEFT_PINKY', 18: 'RIGHT_PINKY',
    19: 'LEFT_INDEX', 20: 'RIGHT_INDEX', 21: 'LEFT_THUMB', 22: 'RIGHT_THUMB',
    23: 'LEFT_HIP', 24: 'RIGHT_HIP', 25: 'LEFT_KNEE', 26: 'RIGHT_KNEE',
    27: 'LEFT_ANKLE', 28: 'RIGHT_ANKLE', 29: 'LEFT_HEEL', 30: 'RIGHT_HEEL',
    31: 'LEFT_FOOT_INDEX', 32: 'RIGHT_FOOT_INDEX'
}

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
                landmarks = results.pose_world_landmarks.landmark
                # 存储当前帧的关键点数据
                frame_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks]
                all_landmarks.append(frame_landmarks)

                counter, status = TypeOfExercise(landmarks).calculate_exercise(
                    exercise_type, counter, status)
            except:
                pass


            if results.pose_world_landmarks:
                for i, landmark in enumerate(results.pose_world_landmarks.landmark):
                    # 获取关键点坐标
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

                    # 绘制关键点中文名称
                    cv2.putText(frame, keypoint_names.get(i, f'P{i}'), (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # 渲染关键点
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_world_landmarks,
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
output_dir = "../../Landmarks/body/Landmarks"
output_path = os.path.join(output_dir, "all_landmarks_3.npy")

# 确保目标文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 保存到指定文件
np.save(output_path, all_landmarks_np)

print(f"关键点数据保存至 {output_path}，共 {len(all_landmarks)} 帧")
