import cv2
import numpy as np

# Завдання 1
# Відкрийте зображення data/Lenna.png. Прочитайте маски
# data/mask1.png та data/mask2.png.
image_lenna = cv2.imread(
    "data/lesson1/Lenna.png",
    cv2.IMREAD_GRAYSCALE
)

image_mask1 = cv2.imread(
    "data/lesson1/mask1.png",
    cv2.IMREAD_GRAYSCALE
)

image_mask2 = cv2.imread(
    "data/lesson1/mask2.png",
    cv2.IMREAD_GRAYSCALE
)

cv2.imshow("leena", image_lenna)
cv2.imshow("mask1", image_mask1)
cv2.imshow("mask2", image_mask2)

# Об’єднайте дві маски в одну, скористайтесь cv2.bitwise_or()
# та виведіть результат
# Виведіть ту частину зображення, яка відповідає:
#  mask1

# mask1 = image_mask1.astype(np.bool)
# print(mask1.shape)
# print(mask1.dtype)
#
# image_lenna[~mask1] = 0
# cv2.imshow("lenna1", image_lenna)

#  mask2

# mask2 = image_mask2.astype(np.bool)
# print(mask2)
# print(mask2.shape)
# print(mask2.dtype)
#
# image_lenna[~mask2] = 0
# cv2.imshow("lenna2", image_lenna)

#  mask1 і mask2

united_mask_im = cv2.bitwise_or(image_mask1, image_mask2)
united_mask = united_mask_im.astype(bool)

image_lenna[~united_mask] = 0
cv2.imshow("lenna3", image_lenna)

# Усі пікселі які не відповідають маскам замінити на 0, перед
# застосуванням змініть тип даних у масці на bool
image_baboo = cv2.imread(
    "data/lesson1/baboo.jpg",
    cv2.IMREAD_GRAYSCALE
)

image_baboo = image_baboo[15:50, 60:195]
cv2.imshow("baboo eyes", image_baboo)


cv2.waitKey(0)
