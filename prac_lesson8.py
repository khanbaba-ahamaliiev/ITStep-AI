import cv2
import numpy as np
from ultralytics import YOLO


# Завдання 1
# Відкрийте зображення data/lesson_seg/crop3.jpg
# Проведіть сегментацію зображення використовуючи
# модель data/lesson_seg/crop-seg.jpg
# Покажіть усі маски рослин з підписами назви цієї
# рослини.
# Покажіть також самі рослини, для цього застосуйте
# маску, і всі зайві пікселі замініть на 255(зробити білий фон)
model = YOLO("data/lesson_seg/crop-seg.pt")

# img = cv2.imread("data/lesson_seg/crop3.jpg")
# cv2.imshow("img", img)
#
# results = model.predict(img)
# result = results[0]
#
# res_img = result.plot()
# cv2.imshow("res_img", res_img)
#
# masks = result.masks
# masks_data = masks.data
# masks_data = masks_data.cpu().numpy()
#
# height,width, colors = img.shape
#
# for i in range(len(masks_data)):
#     mask = masks_data[i]
#     mask = cv2.resize(mask, (width, height))
#     mask = mask.astype(bool)
#
#     new_img = img.copy()
#     new_img[~mask] = 255
#     cv2.imshow(f"plant {i}", new_img)

# Завдання 2
# Відкрийте зображення data/lesson_seg/crop3.jpg
# Проведіть сегментацію зображення
# Порахуйте розмір кожної рослини(площа маски)
# Покажіть найбільшу рослину кожного виду
img = cv2.imread("data/lesson_seg/crop3.jpg")
width, height, colors = img.shape

results = model.predict(img,)
result = results[0]

res_img = result.plot()
cv2.imshow("res_img", res_img)

masks = result.masks
masks_data = masks.data
masks_data = masks_data.cpu().numpy()

masks_area = []
for mask in masks_data:
    mask_area = mask.sum()
    masks_area.append(mask_area)

print(masks_area)

biggest_mask = max(masks_area)
print(biggest_mask)


index = masks_area.index(biggest_mask)

biggest_mask_img = masks_data[index]
biggest_mask_img = cv2.resize(biggest_mask_img, (width, height))
biggest_mask_img = biggest_mask_img.astype(bool)

img[~biggest_mask_img] = 255
cv2.imshow("biggest_mask", img)



cv2.waitKey(0)