import cv2

# по умолчанию читает цветным

image = cv2.imread("data/lesson2/lego.jpg", cv2.IMREAD_COLOR) # чтобы сделать изображение цветным
image = cv2.resize(image, (500, 500))

print(image.dtype)
print(image.shape)

# cv2.imshow("lego", image)
# cv2.waitKey(0)


# формат(цветовые пространства)
# bgr - blue green red
#
# import utils
# utils.lesson2_bgr_range()


# # получить красный канал (правильный способ)
# red = image.copy()
#
# # надо сохранить красный цвет, а все остальные на 0
#
# # синий
# red[:,:,0] = 0
#
# # зеленый
# red[:,:,1] = 0
#
# print(image.shape)
# cv2.imshow("red", red)
# cv2.waitKey(0)


# цветовой формат hsv
# hue(оттенок) - какой именно цвет 0-179ᵒ
# saturation(насыщенность) - насколько цвет насыщен 0-255
# value(яркость) - насколько светлый 0-255

# import utils
# utils.lesson2_hsv_range()


# получить пиксели желтого цвета
cv2.imshow("orig", image)

# пределы для цвета в hsv
# h -- 40 - 80
# s -- 150 - 255
# v -- 150 - 255

lower = (0, 100, 100)
upper = (20, 255, 255)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # перевести с brg на hsv

# получение маски для правильных пикселей

mask = cv2.inRange(hsv, lower, upper)
cv2.imshow("mask", mask)
cv2.waitKey(0)



# lab
# l - яркость
# import utils
# utils.lesson2_lab_range()
