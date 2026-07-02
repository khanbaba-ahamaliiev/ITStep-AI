import cv2
import utils
# избавление от шума
image = cv2.imread("data/lesson3/castello_blurred.png")
# cv2.imshow("orig", image)

noice = utils.add_gaussian_noise(image, 0, 10) # - делает шумным фотку
# cv2.imshow("noice", noice)


# размытие Гаусса - позволяет уменшить шум
gauss = cv2.GaussianBlur(
    noice, # фотка с шумом
    (3, 3), # размер фильтра
    sigmaX=2.5 # насколько важнее становятся дальние пиксели
    )
# cv2.imshow("gauss", gauss)


# двухсторонний фильтр
bilat = cv2.bilateralFilter(
    noice,
    d = 9, # размер фильтра
    sigmaColor=75, # насколько важные пиксели другого цвета
    sigmaSpace=75, # насколько важные являются дальние пиксели
)
# cv2.imshow("bilat", bilat)

# бинаризация

image = cv2.imread("data/lesson3/darken_page.jpg")
cv2.imshow("darken page", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# threshold = 30
#
# mask = gray < threshold
#
# gray[mask] = 0
# gray[~mask] = 255
# cv2.imshow("gray", gray)


# адаптивная бинаризация
res = cv2.adaptiveThreshold(
    gray,
    255, # белый цвет
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # фильтр для подсчета порога Гаус
    cv2.THRESH_BINARY, # просто надо всегда
    15, # размер фильтра
    2 # насколько пиксель должен отличаться от порога
)
cv2.imshow("adaptive", res)

cv2.waitKey(0)