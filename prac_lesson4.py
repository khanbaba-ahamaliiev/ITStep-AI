import cv2

# Завдання 1
# Відкрийте зображення data/lesson3/notes.png.
# image = cv2.imread("data/lesson3/notes.png")
# cv2.imshow("original", image)

# Проведіть наступні дії:
#  проведіть бінарізацію(звичайну та адаптивну)
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray", gray_image)
# threshold = 120
# mask = gray_image < threshold
# gray_image[mask] = 0
# gray_image[~mask] = 255
# cv2.imshow("binar", gray_image)

# res = cv2.adaptiveThreshold(
#     gray_image,
#     255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     11,
#     2
# )
# cv2.imshow("adaptive", res)
#  застосуйте розмиття(гаусове) візьміть ядра 3, 5, 11 та
# sigmaX 0, 2, 10
# gauss = cv2.GaussianBlur(gray_image, (3,3), 1.4)
# cv2.imshow("gauss", gauss)

# bilat = cv2.bilateralFilter(gray_image, 5, 75, 65)
#
# res = cv2.adaptiveThreshold(
#     bilat,
#     255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     7,
#     2
# )

# cv2.imshow("res", res)
#  повторіть бінарізацію, але перед тим застосуйте bilateral
# filter



# # Завдання 2
# # Відкрийте зображення data/lesson3/sudoku.jpg. Проведіть
# # для нього бінарізацію, а саме
# image = cv2.imread("data/lesson3/sudoku.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("original", image)
#
# #  CLAHE
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# clahe = clahe.apply(gray)
# cv2.imshow("clahe", clahe)
#
# #  гаусове розмиття
# gauss = cv2.GaussianBlur(gray, (5,5), 1)
# cv2.imshow("gauss", gauss)
#
# #  адаптивна бінарізація
# result = cv2.adaptiveThreshold(
#     clahe,
#     255,
#     cv2.ADAPTIVE_THRESH_MEAN_C,
#     cv2.THRESH_BINARY,
# 11,
#     2
# )
#
# cv2.imshow("result clahe", result)
#
# result = cv2.adaptiveThreshold(
#     gauss,
#     255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     5,
#     2
# )
# cv2.imshow("result gauss", result)
#  NLMean
# fastmean = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
# cv2.imshow("fastmean", fastmean)
#
# result = cv2.adaptiveThreshold(
#     fastmean,
#     255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     11,
#     2
# )
#
# cv2.imshow("result binary", result)
# Самостійно підберіть параметри, збережіть результат.
# Порівняйте результати для гаусової та середньої адаптивної
# бінарізаці


cv2.waitKey(0)

