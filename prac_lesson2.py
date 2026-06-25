import cv2
import numpy as np
# Завдання 1
# Відкрийте зображення data/Lenna.png. Виведіть на екран
# розмір зображення, тип даних, максимальну та мінімальну
# інтенсивність пікселів, саме зображення з підписом.
image = cv2.imread(
    "data/lesson1/Lenna.png",
    cv2.IMREAD_GRAYSCALE
)
# print(image.shape)
# print(image.dtype)
# print(image.min(), image.max())
cv2.imshow("Lenna", image)
# cv2.waitKey(0)


# Завдання 2
# Відкрийте зображення data/Lenna.png. Виведіть на екран
# такі зображень:
#  Верхній лівий кут розміром 100х50
# cv2.imshow("Lenna", image[:101,:51])

#  Центральний квадрат розміром 100х100
# cv2.imshow("Lenna", image[78:179, 78:179])
#  Верхню половину
# cv2.imshow("upside", image[:128, :])

#  Нижню половину
# cv2.imshow("bottom", image[129:255, :])

#  Ліву половину
# cv2.imshow("leftside", image[:, :128])

#  Праву половину
# cv2.imshow("rightside", image[:, 129:255])



# Завдання 3
# Відкрийте зображення data/Lenna.png. Створіть наступні
# зображення

# image[:20, :] = 0
# image[235:255, :] = 255

# image[:, :20] = 0
# image[:, 235:256] = 0

# image[:40, :] = 0
# image[210:256, :] = 0
# image[:, :40] = 0
# image[:, 215:256] = 0

# cv2.imshow("Lenna", image)

# Завдання 4
# Відкрийте зображення data/Lenna.png. Створіть маску для
# пікселів з інтенсивністю більше 128 та виведіть її. Також
# виведіть заперечення цієї маски.
# На оригінальному зображенні, усі пікселі які не
# відповідають масці замініть на 0 та виведіть результат
mask = image > 128
new_mask = mask.astype(np.uint8)

# cv2.imshow("mask", new_mask * 255)
# print(mask)

image[~mask] = 0
cv2.imshow("Lenna", image)

cv2.waitKey(0)
