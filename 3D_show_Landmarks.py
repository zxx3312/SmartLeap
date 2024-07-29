import numpy as np
import matplotlib.pyplot as plt

keypoints_data = np.load('Landmarks/all_landmarks_1.npy')
A = [11, 12, 15, 16, 23, 24, 25, 26, 27, 28]
txt = ['11', '12', '23', '24', '25', '26', '27', '28']
# 2. 选择第一组数据
first_group = keypoints_data[0]  # assuming data shape is (184, 33, 3)

# 3. 绘制三维图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 提取每个关键点的三维坐标并绘制
for idx, point in enumerate(first_group):
    if idx in A:
        x, y, z = point
        ax.scatter(x, y, z, marker='o')
        ax.text(x,y,z,idx)
    else:
        pass

# 设置图形参数
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图形
plt.title('3D Visualization of First Group Landmarkers')
plt.show()