from ultralytics import YOLO
import cv2

weight_path = r"C:\Users\31825\Desktop\Talkweb\Weight\yolov8l.pt"
car_plate_weight_path = r"C:\Users\31825\Desktop\Talkweb\Weight\plate_detect.pt"

model = YOLO(weight_path)
plate_model = YOLO(car_plate_weight_path, task="detect")

if __name__ == '__main__':
    test_img = cv2.imread(r"C:\Users\31825\Desktop\Talkweb\Multi_count\dataset\image\微信截图_20250224143159.png")
    result = plate_model(test_img)

    res = result[0].plot()
    cv2.namedWindow("a", cv2.WINDOW_NORMAL)
    cv2.imshow("a", res)
    cv2.waitKey()
