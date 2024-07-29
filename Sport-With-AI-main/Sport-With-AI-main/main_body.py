import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
import mediapipe as mp

# 初始化 Mediapipe 相关
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 读取视频文件
cap = cv2.VideoCapture("../../视频简单预处理/Crop_Video/jump_clip_3.mp4")

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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)

        if results.pose_landmarks:
            frame_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in results.pose_world_landmarks.landmark]
            all_landmarks.append(frame_landmarks)

            frame_landmarks = np.array(frame_landmarks)
            x = frame_landmarks[:, 0]
            y = frame_landmarks[:, 1]
            z = frame_landmarks[:, 2]

            # 创建 3D 图形
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # 绘制关键点
            ax.scatter(x, y, z, c='r', marker='o')

            # 绘制每个关键点的名称（可选）
            for i in range(len(x)):
                ax.text(x[i], y[i], z[i], keypoint_names.get(i, f'P{i}'), size=10, zorder=1)

            # 设置坐标轴标签
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

            # 设置图形标题
            plt.title(f'3D Keypoints Visualization (Frame {frame})')

            # 设置视角
            ax.view_init(elev=45, azim=-45)  # elev 设为 90，将 z 轴放在底部

            # 显示图形
            plt.show()

            # 绘制关键点
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                cv2.putText(frame, keypoint_names.get(i, f'P{i}'), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(174, 139, 45), thickness=2, circle_radius=2))

        cv2.imshow('Video', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 转换为Numpy数组并保存
all_landmarks_np = np.array(all_landmarks)
# 设置保存路径
output_dir = "../../Landmarks/body/Crop_Landmarks"
output_path = os.path.join(output_dir, "all_landmarks_3.npy")

# 确保目标文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 保存到指定文件
np.save(output_path, all_landmarks_np)

print(f"关键点数据保存至 {output_path}，共 {len(all_landmarks)} 帧")
