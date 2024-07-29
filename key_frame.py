import os

import cv2
import numpy as np

# 加载保存的关键点数据
keypoints_data = np.load('Landmarks/world/Crop_Landmarks/all_landmarks_3.npy')

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

    last_pre_swing = -1
    last_back_swing = -1
    take_off = -1
    flight = -1
    landing = -1

    pmin_hand_height = 1000
    bmin_hand_height = 1000
    max_height = -1000
    min_height = 1000
    for i in range(num_frames):
        left_heel = keypoints[i][LEFT_HEEL][:2]
        right_heel = keypoints[i][RIGHT_HEEL][:2]
        left_knee = keypoints[i][LEFT_KNEE][:2]
        right_knee = keypoints[i][RIGHT_KNEE][:2]
        left_ankle = keypoints[i][LEFT_ANKLE][:2]
        right_ankle = keypoints[i][RIGHT_ANKLE][:2]
        left_hip = keypoints[i][LEFT_HIP][:2]
        right_hip = keypoints[i][RIGHT_HIP][:2]
        left_shoulder = keypoints[i][LEFT_SHOULDER][:2]
        right_shoulder = keypoints[i][RIGHT_SHOULDER][:2]
        left_wrist = keypoints[i][LEFT_WRIST][:2]
        right_wrist = keypoints[i][RIGHT_WRIST][:2]
        left_toe = keypoints[i][LEFT_TOE][:2]
        right_toe = keypoints[i][RIGHT_TOE][:2]

        hip_center = np.mean([left_hip, right_hip], axis=0)
        shoulder_center = np.mean([left_shoulder, right_shoulder], axis=0)
        heel_height = min(left_heel[1], right_heel[1])
        hand_height = min(right_wrist[1], left_wrist[1])
        left_foot_angle = calculate_angle(left_knee, left_heel, left_toe)
        right_foot_angle = calculate_angle(right_knee, right_heel, right_toe)
        flight_angle = abs(calculate_k(right_shoulder, right_heel))
        hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

        print(i)
        print('heel_height: ', heel_height)
        print('angle: ', right_foot_angle)
        print('left_wrist[0]  left_shoulder[0]: ', left_wrist[0], left_shoulder[0])
        print('hand height: ', hand_height)
        print('k: ', flight_angle)
        print('hip angle: ', hip_angle)

        # 起飞
        if (left_foot_angle > 120 and right_foot_angle > 120) and take_off == -1 and last_back_swing != -1 and flight_angle < 10:
            print('i: ', i)
            take_off = i
        # 前摆: 手向前举到峰值
        elif (left_wrist[0] > left_shoulder[0] and right_wrist[0] > right_shoulder[0]) and (hand_height < pmin_hand_height) and take_off == -1 and (flight_angle > 10):
            last_pre_swing = i
        # 后摆: 手向后举到峰值
        elif (left_wrist[0] < left_hip[0] and right_wrist[0] < right_hip[0]) and (hand_height < bmin_hand_height) and last_pre_swing != -1 and take_off == -1 and hip_angle < 170:
            last_back_swing = i
        # 腾空: 脚后跟离地最高点
        elif heel_height < min_height and take_off != -1:
            print(min_height)
            min_height = heel_height
            flight = i
        # 落地: 脚后跟第一次触地
        elif (left_wrist[1] > left_shoulder[1] and right_wrist[1] > right_shoulder[1]) and (left_wrist[0] > left_shoulder[0] and right_wrist[0] > right_shoulder[0]) and flight != -1:
            print(left_wrist[1], left_shoulder[1], right_wrist[1], right_shoulder[1])
            landing = i
            break

        pmin_hand_height = hand_height
        bmin_hand_height = hand_height

    return last_pre_swing, last_back_swing, take_off, flight, landing

# def get_keyframe_indices(phases):
#     last_pre_swing = None
#     last_back_swing = None
#     take_off = None
#     flight = None
#     landing = None
#
#     for i, phase in enumerate(phases):
#         if phase == '前摆':
#             last_pre_swing = i
#         elif phase == '后摆':
#             last_back_swing = i
#         elif phase == '起飞' and take_off is None:
#             take_off = i
#         elif phase == '腾空' and take_off is not None:
#             flight = i
#         elif phase == '落地' and take_off is not None:
#             landing = i
#             break
#
#     return last_pre_swing, last_back_swing, take_off, flight, landing

# 计算地面高度

# 检测动作阶段
# phases = detect_phases(keypoints_data, ground_level)
#
# # 获取关键帧索引
# keyframe_indices = get_keyframe_indices(phases)

keyframe_indices = detect_phases(keypoints_data)

# 输出关键帧索引
print(f"关键帧索引: {keyframe_indices}")

# 设置视频文件路径
video_source = "视频简单预处理/Crop_Video/jump_clip_3.mp4"
output_dir = "Keyframes/3"
os.makedirs(output_dir, exist_ok=True)

# 读取视频文件
cap = cv2.VideoCapture(video_source)
cap.set(3, 800)  # 设置宽度
cap.set(4, 480)  # 设置高度

# 提取并保存关键帧
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_number in keyframe_indices:
        frame_index = keyframe_indices.index(frame_number)
        filename = os.path.join(output_dir, f"keyframe_{frame_index}.jpg")
        cv2.imwrite(filename, frame)
        print(f"保存关键帧 {filename}")
        if frame is not None:
            cv2.imshow(f"Keyframe: {filename}", frame)
            cv2.waitKey(0)  # 等待键盘输入，以便查看每一帧

    frame_number += 1

cap.release()
cv2.destroyAllWindows()

print(f"提取并保存了 {len(keyframe_indices)} 个关键帧")
