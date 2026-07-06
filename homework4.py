import cv2

# Завдання 1
# Відкрийте зображення data/lesson3/sonet.png. Проведіть
# бінарізацію.
# Обов’язково використайте:
#  розмиття або наведення різкості
#  адаптивну бінарізацію
#  очищеня шумів

image = cv2.imread("data/lesson3/sonet.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("image", image)

gauss = cv2.GaussianBlur(
    image,
    (3,3),
    1.5
)
# cv2.imshow("gauss", gauss)

sharp = cv2.addWeighted(
    image,
    1.5,
    gauss,
    -0.5,
    0
)

# cv2.imshow("sharp", sharp)


result = cv2.adaptiveThreshold(
    gauss,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    7,
    1.5
)

# cv2.imshow("result", result)

# Завдання 2
# Відкрийте зображення data/lesson3/sonnet_noised.png.
# Проведіть бінарізацію. Застосуйте код з завдання 1 та
# спробуйте покращити результат
image = cv2.imread("data/lesson3/sonet_noised.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("original", image)

gauss = cv2.GaussianBlur(
    image,
    (3,3),
    1.5
)
cv2.imshow("gauss", gauss)

# bilat = cv2.bilateralFilter(
#     image,
#     9,
#     75,
#     75
# )

result = cv2.adaptiveThreshold(
    gauss,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2.3
)
cv2.imshow("result", result)

cv2.waitKey(0)