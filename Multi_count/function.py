"""
此代码文件用于实现一些自定义函数如：进行计数、越线等功能的检测
画面帧像素值：
2560 * 1440
左侧车道线坐标：
(550, 470), (1250, 470)
右侧车道线坐标：
(1400, 400), (1850, 400)
"""

import cv2
import numpy as np
import yolov8

left_line = [[400, 470], [1250, 470]]
right_line = [[1300, 400], [2200, 400]]
detect_left_line = [[880, 470], [1250, 470]]

plate_region = [[870, 550], [1200, 570], [1000, 1400], [40, 1200]]


# 传入绘制图片及坐标进行检测框和类别框、类别信息的绘制
def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    # 得到目标矩形框的左上角和右下角坐标
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    # 绘制车辆目标检测框
    cv2.rectangle(image, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)

    # TODO 可以嵌入ocr识别将label替换为车牌检测
    # 根据是否获取到label进行label信息框及相应信息的绘制
    if label:
        # 得到要书写的文本的宽和长，用于给文本绘制背景色
        w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]

        # 文本显示的越界判断
        outside = p1[1] - h >= 3  # int充当bool类型作为越界标志
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # 填充颜色
        # 书写文本
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, 2 / 3, txt_color, thickness=1, lineType=cv2.LINE_AA
                    )


# 绘制预定的检测线
def detect_line(frame_image, color=(0, 0, 255)):
    cv2.line(frame_image, tuple(left_line[0]), tuple(left_line[1]), color=color, thickness=2)
    cv2.line(frame_image, right_line[0], right_line[1], color=color, thickness=2)

    return frame_image


# 过线判断
def pass_line_judge_1(track_list):
    """
    在调用函数前进行长度判断，列表长度 >= 3是进行过线判断，用列表的第一个位置储存bool，意思为是否已经过线
    :param track_list: 用于记录当前id的中心点坐标的列表，采用字典内嵌列表的方式进行存储：{id:[]}
    :return: 根据返回值进行判断，是左侧车道数量+1，还是右侧车道数量+1，还是未出现车辆通过情况
    """
    # 前一时刻和当前时刻的中心点坐标
    # last_x, last_y = track_list[-2]       # 既然能够进入该函数，表明此时list[0]一定为False，那么无需判断[-2]是否未过线
    now_x, now_y = track_list[-1]

    # 根据x坐标值的区间判断是左车道还是右车道
    # 判断是否左侧车道过线
    if left_line[0][0] < now_x < left_line[1][0] and now_y > left_line[0][1]:
        track_list[0] = True
        return 1

    # 右侧车道
    elif right_line[0][0] < now_x < right_line[1][0] and now_y < right_line[0][1]:
        track_list[0] = True
        return 2

    # 都未出现过线的情况
    else:
        return 3


def pass_line_judge_2(track_list):
    """
    该方法用两点进行检测
    :param track_list: 用于记录当前id的中心点坐标的列表，采用字典内嵌列表的方式进行存储：{id:[]}
    :return: 根据返回值进行判断，是左侧车道数量+1，还是右侧车道数量+1，还是未出现车辆通过情况
    """
    # 前一时刻和当前时刻的中心点坐标
    _, last_y = track_list[-2]
    # last_y = float(last_y)
    now_x, now_y = track_list[-1]

    # 根据x坐标值的区间判断是左车道还是右车道
    # 判断是否左侧车道过线
    if left_line[0][0] < now_x < left_line[1][0]:
        if last_y <= left_line[0][1] <= now_y:
            track_list[0] = True
            return 1

    # 右侧车道
    elif right_line[0][0] < now_x < right_line[1][0]:
        if last_y >= right_line[0][1] >= now_y:
            track_list[0] = True
            return 2

    # 都未出现过线的情况
    else:
        return 3


# 基于 过线判断1的字典增添函数
def add_position(position_dict, object_id, position):
    # 没有则创建
    if object_id not in position_dict:
        position_dict[object_id] = [False, False, position]

    else:
        position_dict[object_id].append(position)


def write_num(frame_image, l_num, r_num):
    # 左侧信息写入
    cv2.putText(frame_image, l_num, (900, 450), cv2.FONT_HERSHEY_SIMPLEX, 3,
                (0, 0, 255), 3
                )

    # 右侧信息写入
    cv2.putText(frame_image, r_num, (1625, 380), cv2.FONT_HERSHEY_SIMPLEX, 3,
                (0, 0, 255), 3
                )
    return frame_image


# 绘制视觉上的检测区域
def plate_area(frame_image, color=(0, 255, 0)):
    region = np.array(plate_region, dtype=np.int32)
    cv2.polylines(frame_image, [region], isClosed=True, color=color, thickness=3)
    return frame_image


# 车辆进行车牌检测的逻辑
def plate_area_judge(track_list):
    now_x, now_y = track_list[-1]
    if detect_left_line[0][0] <= now_x <= detect_left_line[1][0] and now_y > detect_left_line[0][1]:
        track_list[1] = True


# 切割汽车区域便于进行车牌检测
def car_split(frame_image, position_list):
    x1, y1, x2, y2 = position_list[:4]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    rectangle_region = frame_image[y1:y2, x1:x2]
    # cv2.imshow("", frame_image)
    # cv2.waitKey()
    return rectangle_region


# 检测车牌并返回车牌区域
# 传入数据应该为裁剪后的车辆图片
def detect_plate_area(car_image):
    plate_detect_result = yolov8.plate_model(car_image)[0]
    location_list = plate_detect_result.boxes.xyxy.tolist()
    location_list = [list(map(int, e)) for e in location_list]
    try:
        a, b, c, d = location_list[0][:4]
        crop_image = car_split(car_image, [a, b, c, d])
        return crop_image
    except:
        print(location_list)
        return None


if __name__ == '__main__':
    test_img = cv2.imread(r"C:\Users\31825\Desktop\Talkweb\Multi_count\dataset\image\4.jpg")
    result = car_split(test_img, [492, 613, 1053, 1111])
    print(result.shape)
    cv2.imshow("", result)

    result = yolov8.plate_model(result)
    result = result[0].plot()

    cv2.namedWindow("a", cv2.WINDOW_NORMAL)
    cv2.imshow("a", result)
    cv2.waitKey(0)
