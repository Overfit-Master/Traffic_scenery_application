import yolov8
import cv2

# 所需文件路径
video_path = r"C:\Users\31825\Desktop\Talkweb\dataset\video\28-1.mp4"

# 初始参数的定义
model = yolov8.model
counter = yolov8.counter

# 读取视频
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "读取视频异常"
# 宽度、高度、帧率
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
print(w, h, fps)

# 检测结果的视频保存器
video_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 以画面帧代替时间
fps_time = 0
past_num = 0
change_num = 0
while cap.isOpened():
    success, im0 = cap.read()
    fps_time += 1
    # 若无法读取到视频帧则表明视频结束，避免死循环
    if not success:
        print("视频处理完成")
        break
    cv2.putText(im0, f"Total:{counter.in_count}", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2)
    # 因为贴信息是一帧一帧进行，所以需要放到循环外，循环内只改变变量
    cv2.putText(im0, f"Traffic flow:{change_num * 60}/min", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2
                )
    while fps_time % 560 == 0:  # 达到2秒时
        fps_time = 0
        if counter.in_count - past_num != 0:
            change_num = counter.in_count - past_num
        past_num = counter.in_count
        break
    # NOTE 此处对源码进行了更改
    im0 = counter.count(im0)
    video_writer.write(im0)
cap.release()
video_writer.release()
cv2.destroyAllWindows()
