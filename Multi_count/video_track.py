from ultralytics import YOLO
import cv2
import yolov8
import function
import ocr_recongnition

weight_path = r"C:\Users\31825\Desktop\Talkweb\Weight\yolov8l.pt"
video_path = r"C:\Users\31825\Desktop\Talkweb\Multi_count\dataset\20\20-2.mp4"

model = YOLO(weight_path).track()

# 读取视频
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "读取视频异常"
# 宽度、高度、帧率
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
print(w, h, fps)

# 检测结果的视频保存器
video_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 定义存储轨迹的字典及左右车道计数器
track_history_dict = {}
left_num = 0
right_num = 0
while cap.isOpened():
    success, im0 = cap.read()
    # 若无法读取到视频帧则表明视频结束，避免死循环
    if not success:
        print("视频处理完成")
        break

    # 绘制过线检测线
    function.detect_line(im0)
    function.plate_area(im0)
    # todo 完成检测和计数
    result = yolov8.model.track(im0, persist=True, device="cuda")
    # 推理在gpu上进行数据格式为GPU Tensor，转为CPU上的int型数据进行统计
    track_ids = result[0].boxes.id.int().cpu().tolist()

    # 遍历一个画面帧的每个检测框数据
    for track_id, box in zip(track_ids, result[0].boxes.data):
        plate_number = ''
        # 绘制及计数部分
        function.box_label(im0, box)
        # 互获取中心点坐标
        x1, y1, x2, y2 = box[:4]
        x = float((x1 + x2) / 2)
        y = float((y1 + y2) / 2)
        # 进行存储
        function.add_position(track_history_dict, track_id, [x, y])

        # 判断是否已经完成过线检测的计数
        # 加入车牌检测后，多了一个bool变量，所以len要大于3
        if not track_history_dict[track_id][0] and len(track_history_dict[track_id]) > 3:
            flag = function.pass_line_judge_2(track_history_dict[track_id])
            if flag == 1:
                left_num += 1

            elif flag == 2:
                right_num += 1

        # # todo 车牌识别部分
        # 先判定是否满足检测条件
        if track_history_dict[track_id][1]:
            car_plate_img = function.car_split(im0, [int(x1), int(y1), int(x2), int(y2)])
            plate_img = function.detect_plate_area(car_plate_img)
            if plate_img is not None:
                plate_number = ocr_recongnition.get_plate_number(plate_img)
            else:
                plate_number = ""
        else:  # 不满足则检测是否在区域中
            function.plate_area_judge(track_history_dict[track_id])

        function.box_label(im0, box, label=str(plate_number))
    function.write_num(im0, str(left_num), str(right_num))
    video_writer.write(im0)
cap.release()
video_writer.release()
cv2.destroyAllWindows()
