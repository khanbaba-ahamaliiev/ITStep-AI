import cv2

# открыть видео
cap = cv2.VideoCapture(
    0, # путь к файлу или 0 для камеры компьютера
)

# информация про видео
# размер кадров
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(width, height)

fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)

# получить первый кадр
success, frame = cap.read()
# success -- True если удалось получить кадр или False
# frame -- само изображение кадр

# cv2.imshow("frame", frame)
# cv2.waitKey(0)


# сохранение видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(
    "result.mp4", # файл куда сохранять
    fourcc, # кодек
    fps, # частота кадрова
    (width, height), # размер (ширина, высота)
    isColor=False # цветное ли изображение
)

# показ видео
while True:
    success, frame = cap.read()

    # проверка получилось ли получить кадр
    if not success:
        break

    # обработка одного файла
    cv2.imshow("camera", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)

    blur = cv2.GaussianBlur(gray, (3,3), 1)
    cv2.imshow("blur", blur)

    binar = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        9,
        2
    )
    cv2.imshow("binary", binar)

    out_writer.write(binar)

    # cv2.waitKey(1) # показывать кадр с задержкой в 1 мс

    # если нажата кнопка то остановить код
    # для esc == 27
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# в конце все закрыть
out_writer.release()
cap.release()


# морфологические операции
# cv2.dilate()
# cv2.erode()