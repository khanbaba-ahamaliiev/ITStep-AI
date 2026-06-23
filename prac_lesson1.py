import numpy as np

# # Завдання 1
# # Створіть масив з числами від 1 до 10. Виведіть його,
# # його розмір, тип даних.
# # Змініть розмір масиву на (5, 2). Знову виведіть масив,
# # розмір та тип даних
# nums = np.arange(1, 11)
# print(nums)
# print(nums.shape)
# print(nums.dtype)
#
# nums = nums.reshape(5, 2)
# print(nums)
# print(nums.shape)
# print(nums.dtype)


# Завдання 2
# Створіть масив:
# Використовуючи індекси виведіть:
# ● число 7
# ● другий рядок
# ● останній стовпчик
# ● праву половину
# ● жовту область
# ● замініть жовту область на -1
# ● зробіть перший стовпчик таким самим як і други


# nums = np.arange(1, 13).reshape(3, 4)
# print(nums)
# print(nums.shape)
# print(nums.dtype)

# print(nums[1,2])
#
# print(nums[1,])
#
# print(nums[:,-1])
#
# print(nums[:, 2:4])
#
# print(nums[1:, 1:3])

# nums[1:, 1:3] = -1
# print(nums)

# nums[:,0] = nums[:, 1]
# print(nums)


# Завдання 3
# У масиві з попереднього завдання створіть маску для
# чисел які більші за 6. З її допомогою
# ● виведіть кількість чисел більших за 6
# ● виведіть самі числа
# ● до кожного числа яке відповідає масці додайте 10
# ● кожне число що не відповідає масці помножте на -1
# ● замініть ці числа які відповідають масці на відповідні
# їм з масиву

# nums = np.arange(1, 13).reshape(3, 4)
# print(nums)
# print(nums.shape)
# print(nums.dtype)
#
# mask = nums > 6
# print(len(mask))
# print(nums[mask])
#
# new_nums = nums
# new_nums[mask] = new_nums[mask] + 10
# print(new_nums)
#
# new_mask = ~mask
# print(new_mask)
# new_nums[new_mask] *= -1
# print(new_nums)
#
# new_aray = np.array(
#     [
#         [1,0,1,0],
#         [0,1,0,1],
#         [1,0,1,0],
#     ]
# )
#
# new_nums[new_mask] = new_aray[new_mask]
# print(new_nums)

# Завдання 6
# Створіть масив типу uint8
# 10 4 25 40 200
# |Помножте всі значення на 2. Результат має бути типу
# uint8 а всі значення в діапазоні 0-255
# Помножте всі значення на 1.5. Результат має бути типу
# uint8 а всі значення в діапазоні 0-255


nums = np.array([10, 4, 25, 40 ,200], dtype=np.uint8)
# print(nums)
# print(nums.shape)
# print(nums.dtype)

nums = nums.astype(np.int64)
nums *= 2
mask = nums > 255
print(mask)
print(nums)
print(nums.dtype)
nums[mask] = 255

nums = nums.astype(np.uint8)
print(nums)
print(nums.dtype)

nums = nums * 1.5
print(mask)
print(nums)
print(nums.dtype)
nums[mask] = 255

nums = nums.astype(np.uint8)
print(nums)
print(nums.dtype)
