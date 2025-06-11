from ultralytics import YOLO, solutions


# Note 视频经过手机剪辑后分辨率发生了改变！
""" 
修改检测区域范围，只检测匝道出口的车流
windows编辑器显示：
画面像素大小：1920*1080
左上标定点坐标：(350, 260)
右上标定点坐标：(930, 260)
右下标定点坐标：(1440, 1080)
左下标定点坐标：(30, 1080)

剪辑后：
整体数值除以2
"""

# 加载模型文件
weight = r"C:\Users\31825\Desktop\Talkweb\Weight\yolov8l.pt"
model = YOLO(weight)
print(model.names)

# 设置检测区域，和自定义使用opencv绘制一样，需要按顺时针或是逆时针将点进行排序
region = [(350, 260), (930, 260), (1440, 1080), (30, 1080)]

# 初始化计数器
# counter = solutions.ObjectCounter(model=model, region=region, show_out=True)
# 根据报错：TypeError: expected str, bytes or os.PathLike object, not YOLO，可知此处应该传入模型路径而不是加载好的模型
counter = solutions.ObjectCounter(
    model=weight, region=region, show_in=False, show_out=False, classes=[2, 4], show=False, device="cuda"
                                  )
