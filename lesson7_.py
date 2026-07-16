import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11s-seg.pt") # - модель yolo которая выделяет пиксели объекта (маски)

# img = cv2.imread("data/lesson_seg/human.jpg")
# cv2.imshow("orig", img)
#
# results = model.predict(img)
# result = results[0]
#
# res_img = result.plot()
# cv2.imshow("result", res_img)
#
#
# # маски объектов
# masks = result.masks
# # print(masks)
#
# masks_data = masks.data
#
# human_mask = masks_data[0]
# human_mask = human_mask.cpu().numpy()
#
# # по умолчанию размер не совпадает с оригинальным изображением
#
#
# human_mask = human_mask.astype(np.uint8)
# human_mask *= 255
#
# human_mask = cv2.resize(human_mask, (600,400))
# cv2.imshow("human_mask", human_mask)

# робота с видео
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# фон нашего видео - должен быть такого же размера
background = cv2.imread("data/lesson4/canal.png")
background = cv2.resize(background,(width,height))


while True:
    success, frame = cap.read()

    if not success:
        break

    cv2.imshow("orig", frame)

    results = model.predict(
        frame,
        device="cpu",
    )

    result = results[0]
    res = result.plot()

    # cv2.imshow("res", res)

    masks = result.masks
    masks_data = masks.data

    # человек с наибольшей вероятностью
    human_mask = masks_data[0]

    human_mask = human_mask.cpu().numpy()
    human_mask = human_mask.astype(np.uint8)
    human_mask *= 255
    human_mask = cv2.resize(human_mask, (width, height))

    # cv2.imshow("mask", human_mask)

    mask = human_mask.astype(bool)

    frame[~mask] = background[~mask]

    cv2.imshow("with background", frame)




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()