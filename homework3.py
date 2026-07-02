import cv2
import numpy as np

# Завдання 1
# Відкрийте зображення data\lesson2\darken.png. Проведіть з
# ним наступні операції, переведіть його в HSV формат та
image = cv2.imread("data/lesson2/darken.png")
cv2.imshow("original", image)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv2.imshow("HSV", hsv_image)

h, s, v = cv2.split(hsv_image)

# обробіть канал Value наступними способами:
#  застосуйте вирівнювання гістограм
new_v = cv2.equalizeHist(v)
new_image = cv2.merge((h, s,new_v))
# cv2.imshow("v hist", new_image)
new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)
cv2.imshow("v hist", image)

#  збільшіть значення десь на 20-50%, оскільки тут
# результат буде типу float32 та явно вийде за межі [0-255]
# застосуйте np.clip(value, 0, 255) та value.astype(np.uint8)

new_v = v * 1.4
new_v = np.clip(new_v, 0, 255)
new_v = new_v.astype(np.uint8)
new_image = cv2.merge((h, s,new_v))
new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)
cv2.imshow("v %", new_image)
# Виведіть результати обох обробок на екран


cv2.waitKey(0)