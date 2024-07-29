import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# 获取get3dFrame.py所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建关键点数据的相对路径
keypoints_data_path = os.path.join(current_dir, 'Landmarks', 'all_landmarks_longjump_3dtest_2.npy')
keypoints_data = np.load(keypoints_data_path)


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

# 初始化图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 设置轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置轴的范围
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# 初始化散点图
scatter = ax.scatter([], [], [])

def init():
    scatter._offsets3d = ([], [], [])
    return scatter,

def update(frame):
    x = keypoints_data[frame, :, 0]
    y = keypoints_data[frame, :, 1]
    z = keypoints_data[frame, :, 2]
    scatter._offsets3d = (x, y, z)
    ax.set_title(f'Frame {frame + 1}')
    return scatter,

# 构建输出动画的相对路径
output_dir = os.path.join(current_dir, 'Sport-With-AI-main', 'Sport-With-AI-main', '3dKeyPoint')
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

# 构建完整的输出文件路径
output_file_path = os.path.join(output_dir, 'keypoints_animation_3dtest_2.gif')


# 创建动画并保存
ani = FuncAnimation(fig, update, frames=range(keypoints_data.shape[0]), init_func=init, blit=True)
ani.save(output_file_path, writer=PillowWriter(fps=10))

# 显示动画
plt.show()
