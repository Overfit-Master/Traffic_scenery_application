from paddleocr import PaddleOCR, draw_ocr
import cv2
import function
import yolov8

cls_model_dir = r"C:\Users\31825\Desktop\Talkweb\Weight\paddleModels\whl\cls\ch_ppocr_mobile_v2.0_cls_infer"
rec_model_dir = r"C:\Users\31825\Desktop\Talkweb\Weight\paddleModels\whl\rec"
ocr = PaddleOCR(use_angle_cls=True, lang="ch", det=False, cls_model_dir=cls_model_dir, rec_model_dir=rec_model_dir)


def get_plate_number(plate_image):
    bool_result = ocr.ocr(plate_image, cls=True)[0]
    if bool_result:
        number, confidence = bool_result[0][1]
        return number


if __name__ == '__main__':
    test_image = cv2.imread(r"C:\Users\31825\Desktop\Talkweb\Multi_count\dataset\image\4.jpg")
    crop_image = function.car_split(test_image, [492, 613, 1053, 1111])
    result = yolov8.plate_model(crop_image)[0]
    print(result)
    # [[138.0249481201172, 415.3832702636719, 262.4264221191406, 452.42840576171875]]-->二维列表
    location_list = result.boxes.xyxy.tolist()
    location_list = [list(map(int, e)) for e in location_list]
    x1, y1, x2, y2 = location_list[0][:4]
    crop_image = function.car_split(crop_image, [x1, y1, x2, y2])
    result_1 = ocr.ocr(crop_image, cls=True)[0]
    if result_1:
        license_name, conf = result_1[0][1]
        print(license_name)
