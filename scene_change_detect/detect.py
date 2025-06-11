import cv2
import numpy as np

# 设定阈值（根据实际情况调整）
MEAN_THRESHOLD = 5.0  # 平均运动幅度阈值
PROPORTION_THRESHOLD = 0.25  # 大运动像素比例阈值
MAGNITUDE_THRESHOLD = 10.0  # 单个像素的运动幅度阈值


def detect_large_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 25 * 240)

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("无法读取视频")
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            cv2.resize(prev_gray, (500, 280)), cv2.resize(next_gray, (500, 280)),
            None,
            pyr_scale=0.5,  # 金字塔缩放因子
            levels=3,  # 金字塔层数
            winsize=15,  # 窗口大小
            iterations=3,  # 迭代次数
            poly_n=5,  # 像素邻域大小
            poly_sigma=1.2,  # 高斯标准差
            flags=0
        )

        # 计算运动幅度和角度
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # print(magnitude, angle)

        # 方法2: 检查大运动像素比例
        large_motion = (magnitude > MAGNITUDE_THRESHOLD)
        proportion = np.mean(large_motion)
        print(proportion)
        # if proportion > PROPORTION_THRESHOLD:
        #     print(f"大运动像素比例: {proportion * 100:.2f}%")

        # 更新前一帧
        prev_gray = next_gray

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_large_motion("801_2025-03-11_09-07.mp4")  # 替换为你的视频路径或0使用摄像头
