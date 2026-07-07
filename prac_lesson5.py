import cv2


# Завдання 1
# Виведіть відео з файлу data\lesson7\text.mp4 на екран та
# збережіть в новий файл.
# Змініть розмір зображення.

# cap = cv2.VideoCapture(
#     r'data\lesson7\text.mp4'
# )
#
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out_writer = cv2.VideoWriter(
#     "result.mp4",  # файл куда сохранять
#     fourcc,  # кодек
#     fps,  # частота кадрова
#     (500,700),  # размер (ширина, высота)
#     isColor=True  # цветное ли изображение
# )

# while True:
#     success, frame = cap.read()
#
#     if not success:
#         break
#
#     frame = cv2.resize(
#         frame,
#         (500,700)
#     )
#     cv2.imshow("frame", frame)
#
#     out_writer.write(frame)
#
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
#
# out_writer.release()
# cap.release()

# Завдання 2
# Відкрийте відео з файлу data\lesson7\text.mp4. Проведіть
# бінарізацію кадрів та збережіть в новий файл.

# cap = cv2.VideoCapture(
#     r'data\lesson7\text.mp4'
# )
#
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out_writer = cv2.VideoWriter(
#     "result.mp4",  # файл куда сохранять
#     fourcc,  # кодек
#     fps,  # частота кадрова
#     (500,700),  # размер (ширина, высотqа)
#     isColor=False  # цветное ли изображение
# )
#
# while True:
#     success, frame = cap.read()
#
#     if not success:
#         break
#
#     frame = cv2.resize(
#         frame,
#         (500,700)
#     )
#     cv2.imshow("frame", frame)
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("gray", gray)
#
#     blur = cv2.GaussianBlur(gray, (3,3), 0)
#
#     res = cv2.adaptiveThreshold(
#         blur,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         7,
#         3
#     )
#     cv2.imshow("res", res)
#
#
#     out_writer.write(res)
#
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
#
# out_writer.release()
# cap.release()


# Завдання 3
# Відкрийте відео з файлу data\lesson7shapes.mp4.
# Проведіть виділення країв на кадрах та збережіть в новий
# файл.

cap = cv2.VideoCapture("data\lesson7\shapes.mp4")


# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out_writer = cv2.VideoWriter(
#     "result.mp4",  # файл куда сохранять
#     fourcc,  # кодек
#     fps,  # частота кадрова
#     (500,700),  # размер (ширина, высота)
#     # isColor=False  # цветное ли изображение
# )

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.resize(frame, (700, 500))

    cv2.imshow("frame", frame)

    # get green color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = (40, 50, 40)
    upper =(80, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)

    cv2.imshow("mask", mask)


    # out_writer.write(frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# out_writer.release()
# cap.release()