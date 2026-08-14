# Завдання 1
# Відкрийте відео data/lesson_pose/squat.mp4
# Ваша задача рахувати кількість присідань.
# Отримайте перший кадр та виділіть основні точки.
# Отримайте координати 3-ох точок ноги
# Визначте кут між цими трьома точками. Скористайтесь
# функцією utils.get_angle(x1, y1, x2, y2, x3, y3) де x2, y2 –
# координати коліна(центральна точка)
# Запустіть відео та добавте на сам кадр кут згинання ніг.
# Визначіть нижню межу кута(якщо людина опустилась
# нижче вважаємо що вона достатньо опустилась) та верхню
# межу кута(якщо людина піднялась вище вважаємо що вона
# достатньо піднялась)
# Добавте кількість присідань та
# кут на кожен кадр.

import cv2
import ultralytics
from utils import get_angle


cap = cv2.VideoCapture("data/lesson_pose/squat.mp4")
model = ultralytics.YOLO("yolo11s-pose.pt")

min_angle = 90
max_angle = 120
count = 0
down = False

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

    results = model.predict(frame, device="cpu")
    result = results[0]

    keypoints = result.keypoints

    xy = keypoints.xy
    xy = xy.cpu().numpy()
    xy = xy[0]
    xy = xy.astype(int)

    x_right_hip, y_right_hip = xy[12]
    x_right_knee, y_right_knee = xy[14]
    x_right_leg, y_right_leg = xy[16]

    cv2.circle(
        frame,
        (x_right_hip, y_right_hip),
        10,
        (0, 255, 0),
        -1
    )

    cv2.circle(
        frame,
        (x_right_knee, y_right_knee),
        10,
        (0, 255, 0),
        -1
    )
    cv2.circle(
        frame,
        (x_right_leg, y_right_leg),
        10,
        (0, 255, 0),
        -1
    )

    angle = get_angle(x_right_hip, y_right_hip, x_right_knee, y_right_knee, x_right_leg, y_right_leg)
    if angle < min_angle:
        down = True

    if angle > max_angle and down:
        count += 1
        down = False

    cv2.putText(
        frame,
        f"Angle: {int(angle)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Count: {count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("gym", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
