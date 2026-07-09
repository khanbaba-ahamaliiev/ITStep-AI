import cv2
import ultralytics


# Завдання 1
# Отримайте перший кадр з файлу data\lesson8\animals.mp4
# та виведіть його на екран.
# Проведіть детекцію об’єктів зо допомогою YOLO та
# виведіть результати.
# Змініть параметри моделі conf та iou і подивіться як це
# впливає на результат.
# Отримайте рамки для кожного об’єкта, виріжіть їх та
# виведіть як окремі зображення

model = ultralytics.YOLO("yolo11s.pt")

cap = cv2.VideoCapture(r"data\lesson8\animals.mp4")

# success, image = cap.read()
# image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
# # cv2.imshow('image', image)
#
# results = model.predict(
#     image,
#     conf=0.5,
#     iou=0.7,
#     device="cpu"
# )
# result = results[0]
#
# res_image = result.plot()
# # cv2.imshow('res_image', res_image)
#
# boxes = result.boxes
#
# for box in boxes:
#     cls = box.cls
#     cls = cls.cpu().numpy()
#
#     conf = box.conf
#     conf = conf.cpu().numpy()
#
#     xyxy = box.xyxy
#     xyxy = xyxy.cpu().numpy()
#     xyxy = xyxy.astype(int)
#     x1, y1, x2, y2 = xyxy[0]
#
#     animal = image[y1:y2, x1:x2]
#     # cv2.imshow(f'animal: {result.names[cls[0]]} - {conf[0] * 100:.2f}%', animal)



while True:
    success, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    if not success:
        break

    results = model.predict(
        frame,
        conf=0.5,
        iou=0.7,
        device="cpu"
    )
    result = results[0]

    res_image = result.plot()


    boxes = result.boxes
    for i in range(len(boxes)):
        box = boxes[i]
        cls = box.cls
        cls = cls.cpu().numpy()

        conf = box.conf
        conf = conf.cpu().numpy()

        xyxy = box.xyxy
        xyxy = xyxy.cpu().numpy()
        xyxy = xyxy.astype(int)
        x1, y1, x2, y2 = xyxy[0]

        animal = frame[y1:y2, x1:x2]
        cv2.imshow(f'animal {i}: {result.names[cls[0]]} ', animal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
