import cv2
import os

def apply_gaussian_blur(video_path, output_path, kernel_size=(15, 15)):
    # 打开输入视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取视频的宽度、高度和帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 定义视频编解码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用'mp4v'编码器
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 对帧应用高斯模糊
        blurred_frame = cv2.GaussianBlur(frame, kernel_size, 0)

        # 写入处理后的帧
        out.write(blurred_frame)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 输入视频的路径
input_video_path = r"Crop_Video/jump_clip_2.mp4"

# 输出视频的保存路径
output_video_path = r"Blurred_Video/blurred_video_2.mp4"
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# 应用高斯模糊并保存视频
apply_gaussian_blur(input_video_path, output_video_path)

print(f"处理后的视频已保存至 {output_video_path}")
