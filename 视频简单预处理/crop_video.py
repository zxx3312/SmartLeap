import os
import cv2
import numpy as np

# 加载保存的关键点数据
keypoints_data = np.load('../Landmarks/body/origin/all_landmarks_2.npy')

# 定义关键点的索引
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_TOE = 31
RIGHT_TOE = 32

def calculate_angle(a, b, c):
    """计算三个点之间的角度"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_k(a, b):
    a = np.array(a)
    b = np.array(b)
    return (a[1] - b[1]) / (a[0] - b[0])

def detect_phases(keypoints):
    num_frames = keypoints.shape[0]

    take_off = -1

    for i in range(num_frames):
        left_knee = keypoints[i][LEFT_KNEE][:2]
        right_knee = keypoints[i][RIGHT_KNEE][:2]
        left_heel = keypoints[i][LEFT_HEEL][:2]
        right_heel = keypoints[i][RIGHT_HEEL][:2]
        left_toe = keypoints[i][LEFT_TOE][:2]
        right_toe = keypoints[i][RIGHT_TOE][:2]
        left_shoulder = keypoints[i][LEFT_SHOULDER][:2]
        right_shoulder = keypoints[i][RIGHT_SHOULDER][:2]
        left_wrist = keypoints[i][LEFT_WRIST][:2]
        right_wrist = keypoints[i][RIGHT_WRIST][:2]

        left_foot_angle = calculate_angle(left_knee, left_heel, left_toe)
        right_foot_angle = calculate_angle(right_knee, right_heel, right_toe)
        flight_angle = abs(calculate_k(right_shoulder, right_heel))
        print(i)
        print('left_foot_angle,right_foot_angle: ', left_foot_angle, right_foot_angle, flight_angle)

        # 起飞
        if flight_angle < 2:
            take_off = i
            break

    return take_off

# 获取起跳帧
take_off_frame = detect_phases(keypoints_data)
print(f"起跳帧索引: {take_off_frame}")

# 设置视频文件路径
video_source = "../videos/2.mp4"
output_video = "Crop_Video/body/jump_clip_2.mp4"
os.makedirs(os.path.dirname(output_video), exist_ok=True)

# 读取视频文件
cap = cv2.VideoCapture(video_source)

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
print(f"视频帧率: {fps}, 总帧数: {frame_count}, 时长: {duration} 秒")

# 计算裁剪的开始帧和结束帧
start_frame = max(take_off_frame - int(2.2 * fps), 0)
end_frame = min(frame_count, take_off_frame + int(3 * fps))
print(f"裁剪开始帧: {start_frame}, 结束帧: {end_frame}")

# 设置输出视频编解码器和帧率
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

# 从开始帧读取并保存视频
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
for i in range(start_frame, end_frame):
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"视频已从起跳帧前3秒裁剪并保存到: {output_video}")
