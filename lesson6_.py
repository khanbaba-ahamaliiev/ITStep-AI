import cv2
import ultralytics
import numpy as np

# создание модели
# модель может одновременно обработатб несколько фотографий [img1, img2, img3]
# на выходе results - это будет список результатов
model = ultralytics.YOLO("yolo11s.pt")

cap = cv2.VideoCapture('data/lesson8/cars+bikes.mp4')

success, image = cap.read()

image = cv2.resize(image, None, fx=0.5, fy=0.5)
cv2.imshow('image', image)


# использование модели
results = model.predict(
    image,
    conf=0.25, # минимальная вероятность для объектов
    iou=0.5, # насколько сильно могут пересекаться объекты
    # classes = [0,1] # все классы которые надо учитывать
)

# получение результата
result = results[0]
# print(results)
# print(type(results))

# получить название классов
names = result.names
# print(names)
# print(type(names))

# сами объекты
boxes = result.boxes
# print(boxes)
# print(type(boxes))

# визуализация результатов
res_img = result.plot()
cv2.imshow('res_img', res_img)

# вероятности
conf = boxes.conf
print(conf)
print(type(conf))

# отключить от графического процессора
conf = conf.cpu()

# перевести в массив numpy
conf = conf.numpy()
# print(conf)
# print(conf.shape)
# print(conf.dtype)

# рамка(boxes)
box = boxes[0] # данные 1 объекта
# print(box.conf)
# print(box.cls)
# print(box.xyxy)

# координаты
xyxy = box.xyxy
# print(xyxy)

# перевод координат в int
xyxy = xyxy.cpu().numpy()
xyxy = xyxy.astype(int)
# print(xyxy)

# вырезать объект из всего изображения
x1, y1, x2, y2 = xyxy[0]

# region of interest
roi = image[y1:y2, x1:x2]

cv2.imshow('roi', roi)




cv2.waitKey(0)