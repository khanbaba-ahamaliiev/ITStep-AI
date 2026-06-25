# чтение изображений
import cv2 # opencv

image = cv2.imread(
    "data/lesson1/cameraman.png",
    cv2.IMREAD_GRAYSCALE # ФЛАГ КАК ЧИТАТЬ ИЗОБРАЖЕНИЕ (ЧЕРНО-БЕЛОЕ)
)
# print(type(image))
# print(image.shape)
# print(image.dtype)
# print(image)


# показать изображение

# cv2.imshow(
#     "original", # название
#     image # изображение
# )

# чтоб изображение показывалось долго надо зациклить програму
# ждем пока не будет нажата какая-то кнопка
# cv2.waitKey(0)

# изменение размера изображения
# new_image = cv2.resize(image,(500, 500))
# cv2.imshow("name",new_image)
# cv2.waitKey(0)
# print(new_image.shape)

# в пропорциях
# new_image = cv2.resize(image, None, fx=2, fy=2)
# cv2.imshow("name",new_image)
# cv2.waitKey(0)


# вырезали сегмент изображения
segment = image[75:175, 20:240]
print(segment.shape)
print(segment.dtype)


# увеличим значение пикселей
image[75:175, 20:240] += 5


cv2.imshow("image", image)
cv2.imshow("segment", segment)
cv2.waitKey(0)


