import numpy as np
import cv2
import matplotlib.pyplot as plt

# 加载保存的关键点数据
keypoints_data = np.load('Landmarks/all_landmarks.npy')

# 设置关键点名称和对应索引
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

# 读取视频文件
video_source = "videos/long-jump.mp4"
cap = cv2.VideoCapture(video_source)

# 获取视频总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def plot_frame_with_projection(frame_index):
    if frame_index < 0 or frame_index >= keypoints_data.shape[0]:
        print(f"Frame index {frame_index} out of range. Please select a value between 0 and {keypoints_data.shape[0] - 1}.")
        return
    
    # 设置视频帧位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {frame_index}")
        return
    
    # 初始化图形
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制原图像
    ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f'Original Frame {frame_index + 1}')
    ax[0].axis('off')
    
    # 获取特定帧的关键点坐标
    x = keypoints_data[frame_index, :, 0]
    y = keypoints_data[frame_index, :, 1]
    
    # 绘制2D投影图像
    ax[1].scatter(x, y)
    
    # # 添加关键点名称
    # for i, (xi, yi) in enumerate(zip(x, y)):
    #     ax[1].text(xi, yi, keypoint_names.get(i, f'P{i}'), fontsize=9)
    
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([1, 0])  # Invert y-axis to match image coordinates
    ax[1].set_title(f'2D Keypoints Projection Frame {frame_index + 1}')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    
    plt.show()

# 示例：绘制第10帧图像和2D关键点投影
plot_frame_with_projection(157)
plot_frame_with_projection(151)
plot_frame_with_projection(158)
plot_frame_with_projection(160)
plot_frame_with_projection(173)
# 释放视频捕获对象
cap.release()
