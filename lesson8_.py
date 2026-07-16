import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO

# model = YOLO("yolo11s-seg.pt") # - модель yolo которая выделяет пиксели объекта (маски)
#
#
# img = cv2.imread("data/lesson_seg/human.jpg")
#
# cv2.imshow("orig", img)
#
# results = model.predict(img)
# result = results[0]
#
# res = result.plot()
# cv2.imshow("res", res)
#
# masks = result.masks
# print(masks)
#
# masks_data = masks.data
# masks_data = masks_data.cpu().numpy()
#
# mask3 = masks_data[2]
#
# height, width, color = img.shape
# mask3 = cv2.resize(mask3, (width, height))
#
# mask3_bool = mask3.astype(bool)
# mask3_uint = mask3.astype(np.uint8)
# mask3_uint *= 255
#
#
# img[~mask3_bool] = 0
#
# cv2.imshow("mask3", img)



# обучение модели
model = ultralytics.YOLO("yolo11s.pt")

model.train(
    data="data/yolo-dataset/dataset.yaml",
    device="cpu",
    batch=6, # количество изображений которых модель видит за 1 раз
)



cv2.waitKey(0)