import os
import cv2
import numpy as np

# 加载保存的关键点数据
keypoints_data = np.load('../Landmarks/world/Crop_Landmarks/all_landmarks_3.npy')


def extract_frames_with_keypoints(video_path, save_path, keypoints_data):
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    total_frames = keypoints_data.shape[0]

    while True:
        success, frame = video.read()
        if not success or frame_count >= total_frames:
            break

        # 获取当前帧的关键点
        keypoints = keypoints_data[frame_count]

        # 绘制关键点及其坐标
        for idx, (x, y, z) in enumerate(keypoints):
            cx, cy = int(x * frame.shape[1]), int(y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{idx} ({cx}, {cy})", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 保存帧图像
        frame_save_path = f'{save_path}/frame_{frame_count}.jpg'
        cv2.imwrite(frame_save_path, frame)
        frame_count += 1

    video.release()
    cv2.destroyAllWindows()


# 替换为实际视频文件的路径
video_path = r"Crop_Video/jump_clip_3.mp4"

# 替换为想要保存帧图片的目录路径
save_path = r"../all_frames_with_keypoints"
os.makedirs(save_path, exist_ok=True)

# 提取并保存带关键点的帧图像
extract_frames_with_keypoints(video_path, save_path, keypoints_data)