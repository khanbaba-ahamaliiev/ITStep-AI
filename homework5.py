import cv2

# Завдання 1
# Відкрийте відео з файлу data\lesson7\meter.mp4.
# Проведіть бінарізацію кадрів та збережіть в новий файл.
# Можливо очистіть від шуму або наведіть різкість через
# bilateralFilter

cap = cv2.VideoCapture("data\lesson7\meter.mp4")

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

    frame = cv2.resize(frame, (500, 850))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)

    # blur = cv2.GaussianBlur(frame, (3,3), 1)

    bilat = cv2.bilateralFilter(
        frame,
        9,
        75,
        75)

    res = cv2.adaptiveThreshold(
        bilat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        13,
        2
    )
    cv2.imshow('binar', res)

    # out_writer.write(res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# out_writer.release()
# cap.release()
