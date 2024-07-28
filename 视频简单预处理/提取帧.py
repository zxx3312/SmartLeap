import cv2

"""""
input:将video_path替换为实际的待提取的视频文件的路径
      将save_path替换为提取到的文件夹地址（需要提前新建文件夹）

output：视频的每一帧图像
"""""


def extract_frames(video_path, save_path):
    video = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        # cv2.imshow('Frame', frame)
        # cv2.waitKey(1)

        frame_save_path = f'{save_path}/frame_{frame_count}.jpg'
        cv2.imwrite(frame_save_path, frame)
        frame_count += 1

    video.release()
    cv2.destroyAllWindows()


# 替换为实际视频文件的路径
video_path = r"D:\Users\86195\Documents\WeChat Files\wxid_jfvo1zhvfh9h12\FileStorage\Video\2024-07\6eeb4dac63b0ece460c1dab48407bc9a.mp4"

# 替换为想要保存帧图片的目录路径
save_path = r"D:\Users\86195\Desktop\peng1_every_frame"

extract_frames(video_path, save_path)
