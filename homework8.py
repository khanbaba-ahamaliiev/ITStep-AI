import cv2
import numpy as np
import ultralytics


# Завдання 1
# Відкрийте зображення data/lesson_seg/tumor1.jpg
# Проведіть сегментацію зображення використовуючи
# модель data/lesson_seg/brain-tumor-seg.jpg
# Визначте площу пухлини в пікселях.
# Визначте площу в
# (1 піксель – 0,0025
# )
# В залежності від площі присвойте пухлині певний тип
#  <10 – small
#  10-25 – middle
#  >25 – large
# Покажіть пухлину – за допомогою маски усі лишні
# пікселі зробіть 0, а як назву зображення використайте її тип

model = ultralytics.YOLO("data/lesson_seg/brain-tumor-seg.pt")

image = cv2.imread("data/lesson_seg/tumor1.jpg")
cv2.imshow("tumor", image)

width, height, colours = image.shape

results = model.predict(image)
result = results[0]

res_img = result.plot()
cv2.imshow("res_img", res_img)

masks = result.masks
masks_data = masks.data
masks_data = masks_data.cpu().numpy()

mask = masks_data[0]
mask_uint = mask.astype(np.uint8)
mask_uint *= 255
cv2.imshow("mask", mask_uint)

mask_bool = mask.astype(bool)
mask_area = mask_bool.sum()
print(f"Площа пухлини в пікселях: {mask_area}")

mask_area_sm = mask_area * 0.0025
print(f"Площа пухлини в см: {mask_area_sm}")

if mask_area < 10:
    print("small")
elif mask_area < 25:
    print("middle")
else:
    print("large")

mask = cv2.resize(mask, (width, height))
mask = mask.astype(bool)

image[~mask] *= 0
cv2.imshow("tumor_with_mask", image)



cv2.waitKey(0)


