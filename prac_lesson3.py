import cv2


# Завдання 1
# Відкрийте зображення data/lesson2/marbles.png.
# Використайте кольорову сегментацію для отримання масок до
# кульок:
#  синього кольору
image = cv2.imread("data/lesson2/marbles.png")
# cv2.imshow("marbles", image)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower = (100, 100, 120)
upper = (130, 255, 255)

mask_blue = cv2.inRange(hsv, lower, upper)
# cv2.imshow("blue", mask_blue)

#  зеленого і червоного
lower = (0, 150, 170)
upper = (7, 255, 255)

mask_red = cv2.inRange(hsv, lower, upper)
# cv2.imshow("mask red", mask_red)

lower = (40, 90, 90)
upper = (80, 255, 255)

mask_green = cv2.inRange(hsv, lower, upper)
# cv2.imshow("mask green", mask_green)

mask_green_red = cv2.bitwise_or(mask_green, mask_red)
# cv2.imshow("mask red green", mask_green_red)
#  чорного
lower = (0, 0, 0)
upper = (180, 100, 50)

mask_black = cv2.inRange(hsv, lower, upper)
# cv2.imshow("mask black", mask_black)

#  білого
lower = (0, 0, 200)
upper = (180, 40, 255)

mask_white = cv2.inRange(hsv, lower, upper)
# cv2.imshow("mask white", mask_white)
#  усіх кульок
mask_rgb = cv2.bitwise_or(mask_green_red, mask_blue)
mask_black_white = cv2.bitwise_or(mask_black, mask_white)
mask_all_color = cv2.bitwise_or(mask_rgb, mask_black_white)

# cv2.imshow("mask", mask_all_color)
cv2.waitKey(0)


# Завдання 2
# Відкрийте зображення data/lesson2/cell.png. Покращте
# зображення за допомогою вирівнювання гістограми. Оскільки
# зображення кольорове, вам доведеться зробити наступні
# кроки:
image = cv2.imread("data/lesson2/cell.png")
cv2.imshow("cell", image)
#  перевести зображення в LAB
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
print(lab.shape)

#  розбити зображення на канали l, a та b
l, a, b = cv2.split(lab)

#  вирівняти гістограму для l
new_l = cv2.equalizeHist(l)


#  зібрати канали назад в зображення
new_lab = cv2.merge((new_l, a, b))
cv2.imshow("new cell", new_lab)

#  перевести результат назад в BGR
bgr = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)
cv2.imshow("bgr", bgr)

# Порівняйте результати для 2 алгоритмів
cv2.waitKey(0)

