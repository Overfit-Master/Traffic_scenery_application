import cv2
import numpy as np
import sys
from datetime import datetime


def calc_optical_flow_farneback(prev_img, next_img, pyr_scale=0.5, levels=3, winsize=15,
                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0):
    # 转换为灰度图像（若输入为彩色）
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY) if len(prev_img.shape) == 3 else prev_img
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY) if len(next_img.shape) == 3 else next_img

    prev_gray = cv2.resize(prev_gray, (600, 340))
    next_gray = cv2.resize(next_gray, (600, 340))

    # 调用OpenCV的Farneback光流函数
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=next_gray,
        flow=None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags
    )
    return flow


# 量化flow指标为运动比例
def calculate_motion_ratio(flow, method='energy', threshold=1.0):
    """
    量化整体画面运动比例
    :param flow: 光流矩阵 (H,W,2)
    :param method: 量化方法 ['energy', 'mean', 'percentage']
    :param threshold: 运动判定阈值（像素）
    :return: 运动比例 (0.0~1.0)
    """
    # 计算各像素运动幅度
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    if method == 'energy':
        # 总运动能量法：所有像素位移的平方和
        total_energy = np.sum(magnitude ** 2)
        max_energy = magnitude.size * (threshold ** 2)  # 基于阈值的理论最大值
        return min(total_energy / max_energy, 1.0)

    elif method == 'mean':
        # 平均位移法：整体平均位移比例
        avg_magnitude = np.mean(magnitude)
        return min(avg_magnitude / threshold, 1.0)

    elif method == 'percentage':
        # 运动像素占比法：超过阈值的像素比例
        moving_pixels = np.sum(magnitude > threshold)
        return moving_pixels / magnitude.size

    else:
        raise ValueError("Invalid method. Choose from 'energy', 'mean', 'percentage'")


# 示例用法
if __name__ == "__main__":
    rtsp_url = "801_2025-03-11_09-07.mp4"
    capture = cv2.VideoCapture(rtsp_url)
    ret, prev_frame = capture.read()

    # 存储历史帧
    frame_buffer = [prev_frame]
    buffer_size = 75

    if not ret:
        print("Video read fail!")
        sys.exit(1)

    # 判断VideoWriter是否创建，避免重复创建报错
    video_writer_flag = False
    # 判断逻辑触发时刻的前75帧图片是否完成存储
    old_save_flag = False
    # 视频写入器
    video_writer = None
    # 是否保存未运动帧
    saving_flag = False
    # 保存未运动帧的数量
    later_frame_num = 0

    while True:
        ret, next_frame = capture.read()
        if not ret:
            print("Video read fail!")
            break

        frame_buffer.append(next_frame)
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)

        flow = calc_optical_flow_farneback(prev_frame, next_frame)
        percent = calculate_motion_ratio(flow, method="percentage")
        percent = round(percent, 3)

        if percent > 0.4:
            print("Scenery change has been detected!")
            # 按照flag变量生成VideoWriter
            if not video_writer_flag:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = capture.get(cv2.CAP_PROP_FPS)
                frame_size = (next_frame.shape[1], next_frame.shape[0])
                video_writer = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)
                video_writer_flag = True

            if not old_save_flag:
                for frame in frame_buffer:
                    video_writer.write(frame)
                old_save_flag = True
            # 当移动比例不再触发时，仍然存储后75帧
            later_frame_num = 75
            saving_flag = True
        else:
            if saving_flag:
                if later_frame_num > 0:
                    video_writer.write(next_frame)
                    later_frame_num -= 1
                else:
                    video_writer.release()
                    video_writer_flag = False
                    old_save_flag = False

        prev_frame = next_frame
    if video_writer is not None:
        video_writer.release()
    capture.release()

