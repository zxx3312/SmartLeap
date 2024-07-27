import numpy as np

# 加载保存的关键点数据
keypoints_data = np.load('Landmarks/all_landmarks.npy')

# 定义关键点的索引
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


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


def detect_phases(keypoints):
    num_frames = keypoints.shape[0]
    phases = [''] * num_frames

    for i in range(num_frames):
        left_ankle = keypoints[i][LEFT_ANKLE][:2]
        right_ankle = keypoints[i][RIGHT_ANKLE][:2]
        left_knee = keypoints[i][LEFT_KNEE][:2]
        right_knee = keypoints[i][RIGHT_KNEE][:2]
        left_hip = keypoints[i][LEFT_HIP][:2]
        right_hip = keypoints[i][RIGHT_HIP][:2]
        left_shoulder = keypoints[i][LEFT_SHOULDER][:2]
        right_shoulder = keypoints[i][RIGHT_SHOULDER][:2]

        # 计算双肩和双髋的中点
        hip_center = np.mean([left_hip, right_hip], axis=0)
        shoulder_center = np.mean([left_shoulder, right_shoulder], axis=0)

        # 计算脚踝高度
        ankle_height = np.mean([left_ankle[1], right_ankle[1]])

        # 前摆: 肩部高于髋部
        if shoulder_center[1] < hip_center[1]:
            phases[i] = '前摆'
        # 后摆: 肩部低于髋部
        elif shoulder_center[1] > hip_center[1]:
            phases[i] = '后摆'
        # 起飞: 脚踝靠近地面，且髋部与肩部的角度接近180度
        elif ankle_height < hip_center[1] and 170 <= calculate_angle(left_hip, shoulder_center, right_hip) <= 190:
            phases[i] = '起飞'
        # 腾空: 脚踝和膝盖高度相近
        elif abs(left_ankle[1] - left_knee[1]) < 0.1 and abs(right_ankle[1] - right_knee[1]) < 0.1:
            phases[i] = '腾空'
        # 落地: 脚踝高度降低
        else:
            phases[i] = '落地'

    return phases


def get_keyframe_indices(phases):
    last_pre_swing = None
    last_back_swing = None
    take_off = None
    flight = None
    landing = None

    for i, phase in enumerate(phases):
        if phase == '前摆':
            last_pre_swing = i
        elif phase == '后摆':
            last_back_swing = i
        elif phase == '起飞' and take_off is None:
            take_off = i
        elif phase == '腾空' and take_off is not None:
            flight = i
        elif phase == '落地' and take_off is not None:
            landing = i
            break

    return last_pre_swing, last_back_swing, take_off, flight, landing


# 检测动作阶段
phases = detect_phases(keypoints_data)

# 获取关键帧索引
keyframe_indices = get_keyframe_indices(phases)

# 输出关键帧索引
print(f"关键帧索引: {keyframe_indices}")
