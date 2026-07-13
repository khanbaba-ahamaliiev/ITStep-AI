import cv2
import ultralytics


#Завдання 1
# Відкрийте відео з файлу data\lesson8\meetings.mp4
# Застосуйте детекцію та виведіть результат, підберіть
# параметри
# Можете змінити розмір кадру для кращої візуалізації
# cv2.resize()
model = ultralytics.YOLO("yolo11s.pt")
cap = cv2.VideoCapture("data\lesson8\meetings.mp4")

# success, frame = cap.read()
# frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
# # cv2.imshow("frame",frame)
#
# results = model.predict(
#     frame,
#     conf=0.60,
#     iou=0.5,
# )
# result = results[0]
# res_image = result.plot()
# # cv2.imshow("res_image", res_image)
#
# names = result.names
# # print(names)
# boxes = result.boxes
# # print(boxes)
# cv2.waitKey(0)

# Завдання 2
# Відкрийте відео з файлу data\lesson8\meetings.mp4
# Застосуйте детекцію та почніть показувати відео з
# моменту, коли людей стало 5

person_come = False

while True:
    success, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)

    if not success:
        break

    results = model.predict(
        frame,
        conf=0.60,
        iou=0.80,
        device="cpu",
        classes=[0]
    )
    result = results[0]

    boxes = result.boxes

    if len(boxes) <= 4 and person_come == False:
        continue

    if len(boxes) > 4:
        person_come = True


    res_image = result.plot()
    cv2.imshow("frame", res_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break