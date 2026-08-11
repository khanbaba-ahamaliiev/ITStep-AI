import cv2
import ultralytics

# Завдання 1
# Відкрийте відео data/lesson_pose/sitting.mp4
# Отримайте перший кадр
# Покажіть його, за потреби змініть розмір

cap = cv2.VideoCapture("data/lesson_pose/sitting.mp4")

success, frame = cap.read()
cv2.imshow("frame", frame)

# Завдання 2
# Застосуйте модель YOLO Pose
# Отримайте результати (result) та виведіть їх на екран
# Використайте параметри device

model = ultralytics.YOLO("yolo11s-pose.pt")

results = model.predict(frame, device="cpu", conf=0.50)
result = results[0]

# Завдання 3
# Користуючись методом plot() отримайте зображення з
# рамками та підписами і покажіть його.

res_image = result.plot()
# cv2.imshow("res_image", res_image)


# Завдання 4
# ● Отримайте інформацію про ключові точки(keypoints)
# ● Виведіть її на екран
# ● Отримайте координати точок(xy)
# ● Виведіть координати на екран разом з типом даних та
# розміром(позбудьтесь тензорів за допомогою cpu() та
# numpy())


keypoints = result.keypoints
# print(keypoints)

xy = keypoints.xy
xy = xy.cpu().numpy()
xy = xy[0]
xy = xy.astype(int)
print(xy)

# Завдання 5
# ● Отримайте координати для лівого коліна, лівої руки,
# правої руки для першого об’єкта
# ● Намалюйте ці точки на зображенні:
# ○ ліве коліно – зелений
# ○ ліва рука – червоний
# ○ права рука – білий
x_left_knee, y_left_knee = xy[13]
cv2.circle(
    frame,
    (x_left_knee, y_left_knee),
    10,
    (0, 255, 0),
    -1
)

x_left_hand, y_left_hand = xy[9]
cv2.circle(
    frame,
    (x_left_hand, y_left_hand),
    10,
    (0, 0, 255),
    -1
)

x_right_hand, y_right_hand = xy[10]
cv2.circle(
    frame,
    (x_right_hand, y_right_hand),
    10,
    (255, 255, 255),
    -1
)

# cv2.imshow("image with circle", frame)

# Завдання 6
# Для кожного кадру на відео намалюйте координати для
# лівого коліна, лівої руки, правої руки
# Беріть координати для першого об’єкта

# while True:
#     success, frame = cap.read()
#
#     if not success:
#         break
#
#     results = model.predict(frame, device="cpu", conf=0.50)
#     result = results[0]
#
#     keypoints = result.keypoints
#
#     xy = keypoints.xy
#     xy = xy.cpu().numpy()
#     xy = xy[0]
#     xy = xy.astype(int)
#
#     x_left_knee, y_left_knee = xy[13]
#     cv2.circle(
#         frame,
#         (x_left_knee, y_left_knee),
#         10,
#         (0, 255, 0),
#         -1
#     )
#
#     x_left_hand, y_left_hand = xy[9]
#     cv2.circle(
#         frame,
#         (x_left_hand, y_left_hand),
#         10,
#         (0, 0, 255),
#         -1
#     )
#
#     x_right_hand, y_right_hand = xy[10]
#     cv2.circle(
#         frame,
#         (x_right_hand, y_right_hand),
#         10,
#         (255, 255, 255),
#         -1
#     )
#
#     cv2.imshow("video with circle", frame)
#
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# Завдання 7
# Під час відео обраховуйте кількість присідань.
# Вважайте що людина присіла якщо рука опустилась
# нижче коліна.
# Кількість присідань відображайте на кадрі(cv2.putText)
# Завдання 8
# Модифікуйте код щоб кількість присідань виводилась
# правильно. Для цього вам потрібно визначати чи людина
# зараз присідає чи піднімається за правилом:
# ● якщо рука нижче коліна то людина встає
# ● якщо рука вище коліна – присідає
# Рахуйте лише ті присідання які відбулись коли людина
# присідає та рука опинилась нижче коліна.
# Разом з кількістю присідань відображайте чи людина
# присідає чи встає


stepup = 0
is_standup = True

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model.predict(frame, device="cpu", conf=0.50)
    result = results[0]
    result_img = result.plot()
    cv2.imshow("video with circle and result", result_img)

    keypoints = result.keypoints
    xy = keypoints.xy
    xy = xy.cpu().numpy()
    xy = xy[0]
    xy = xy.astype(int)

    x_left_knee, y_left_knee = xy[13]
    cv2.circle(
        frame,
        (x_left_knee, y_left_knee),
        10,
        (0, 255, 0),
        -1
    )

    x_left_hand, y_left_hand = xy[9]
    cv2.circle(
        frame,
        (x_left_hand, y_left_hand),
        10,
        (0, 0, 255),
        -1
    )

    x_right_hand, y_right_hand = xy[10]
    cv2.circle(
        frame,
        (x_right_hand, y_right_hand),
        10,
        (255, 255, 255),
        -1
    )

    if y_left_hand > y_left_knee and is_standup:
        stepup += 1
        is_standup = False

    if y_left_hand < y_left_knee:
        is_standup = True

    cv2.putText(
        frame,
        f"stepup: {stepup}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        'standup' if is_standup else 'stepdown',
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.imshow("video with circle", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# Завдання 9
# ● Отримайте 258 кадр з відео
# ● Застосуйте модель
# ● Отримайте результати(result)
# ● Отримайте дані про рамки(boxes)
# ● Отримайте дані про рамку для першого об’єкта та
# виведіть їх
# ● Відобразіть результати(метод plot())
# ● Зробіть висновки





# Завдання 10
# Створіть функцію get_box_area(box)
# Параметри:
# ● box – інформація про рамку об’єкта
# Функція повинна
# ● отримати дані про рамки у форматі xywh
# ● позбудьтесь тензорів
# ● отримайте ширину(w) та висоту(h) рамки
# ● поверніть площу рамки
# За допомогою функцій get_box_area покажіть площі
# рамок для кожного об’єкта на 200 кадрі
# Завдання 11
# Створіть функцію get_largets_box_id(boxes)
# Параметри:
# ● boxes – інформація про рамки всіх об’єктів
# Функція повинна індекс рамки з найбільшою площею.
# Скористайтесь функцією get_box_area
# Завдання 12
# Модифікуйте завдання 8 так щоб вибиралися точки
# об’єкта з найбільшою рамкою.
# Скористайтесь get_largets_box_id(boxes)
# Завдання 13
# Відкрийте відео data/lesson_pose/hopak.mp4
# Покажіть відео добавляючи на кожен кадр назву руху



cv2.waitKey(0)
