import numpy as np
import cv2

# 示例：假设每一帧的人体关键点数据为 Numpy 数组，形状为 (num_frames, num_keypoints, 2)
# num_keypoints 是关键点的数量（如肩膀、膝盖、脚踝等），2 表示 (x, y) 坐标
keypoints_data = np.load('keypoints_data.npy')  # 加载已经提取的关键点数据

import numpy as np

# 假设关键点数据格式如下
# keypoints_data = (num_frames, num_keypoints, 2)
keypoints_data = np.load('keypoints_data.npy')

# 定义关键点的索引
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

def detect_phase(keypoints):
    phases = []
    for frame in keypoints:
        left_knee_y = frame[LEFT_KNEE][1]
        right_knee_y = frame[RIGHT_KNEE][1]
        left_ankle_y = frame[LEFT_ANKLE][1]
        right_ankle_y = frame[RIGHT_ANKLE][1]
        left_shoulder_y = frame[LEFT_SHOULDER][1]
        right_shoulder_y = frame[RIGHT_SHOULDER][1]
        left_hip_y = frame[LEFT_HIP][1]
        right_hip_y = frame[RIGHT_HIP][1]

        # 使用膝盖和脚踝的y坐标检测动作阶段
        if (left_ankle_y - left_knee_y < -20) or (right_ankle_y - right_knee_y < -20):
            phases.append('起飞')
        elif (left_ankle_y - left_knee_y > 20) or (right_ankle_y - right_knee_y > 20):
            phases.append('落地')
        elif abs(left_ankle_y - left_knee_y) < 10 and abs(right_ankle_y - right_knee_y) < 10:
            phases.append('腾空')
        elif (left_shoulder_y - left_hip_y > 20) or (right_shoulder_y - right_hip_y > 20):
            phases.append('前摆')
        else:
            phases.append('后摆')

    return phases

def get_keyframes(phases):
    keyframes_indices = []
    for i, phase in enumerate(phases):
        if phase in ['前摆', '后摆', '起飞', '腾空', '落地']:
            keyframes_indices.append(i)
    return keyframes_indices

# 检测动作阶段
phases = detect_phase(keypoints_data)

# 获取关键帧索引
keyframes_indices = get_keyframes(phases)

# 输出关键帧索引
print(f"关键帧索引: {keyframes_indices}")

# 如果需要保存关键帧图像，请加载原始视频并提取对应帧
# 示例：
video_path = 'video_with_keypoints.mp4'
cap = cv2.VideoCapture(video_path)
for idx in keyframes_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        output_path = f"keyframe_{idx}.jpg"
        cv2.imwrite(output_path, frame)
cap.release()
cv2.destroyAllWindows()
