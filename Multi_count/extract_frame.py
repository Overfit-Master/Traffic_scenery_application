"""
此代码用于抽取视频帧率形成图片，实现单张检测以及代码的测试
"""

import cv2
from PIL import Image
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\31825\Desktop\Talkweb\Multi_count\dataset\20\20-2.mp4")  # 获取视频对象
isOpened = cap.isOpened  # 判断是否打开
# 视频信息获取
fps = cap.get(cv2.CAP_PROP_FPS)

imageNum = 0
image_num = 0
timef = 100  # 隔15帧保存一张图片

while isOpened:

    image_num += 1

    (frameState, frame) = cap.read()  # 记录每帧及获取状态

    if frameState is True and (image_num % timef == 0):

        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))

        frame = np.array(frame)

        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        imageNum = imageNum + 1
        fileName = r'C:\\Users\\31825\\Desktop\\Talkweb\\Multi_count\\dataset\\image\\' + str(imageNum) + '.jpg'  # 存储路径
        cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(fileName + " successfully write in")  # 输出存储状态

    elif not frameState:
        break

print('finish!')
cap.release()
