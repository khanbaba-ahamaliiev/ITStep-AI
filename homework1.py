import numpy as np

# Створіть масив:
# 1 2 3 4
# 5 6 7 8
# 9 10 11 12
# 13 14 15 16

nums = np.arange(1, 17).reshape(4,4)
print(nums)

# Використовуючи індекси виведіть:
# ● число 14
print(nums[-1,1])

# ● третій рядок
print(nums[2,])

# ● перший стовпчик
print(nums[:,0])

# ● верхню половину
print(nums[0:2,:])

# ● замініть числа в рядках 2-3 на 100
nums[1:3] = 100
print(nums)

# ● зробіть другий рядок таким як останній рядок
nums[1] = nums[-1]
print(nums)


# Завдання 2
# У масиві з попереднього завдання створіть маску для
# парних чисел.
mask = nums % 2 == 0
print(mask)

# З її допомогою
# ● виведіть самі числа
print(nums[mask])

# ● замініть їх на 100

nums[mask] = 100
print(nums)


# Завдання 3
# Створіть 2 масиви типу uint8:
# Масив 1: 128 200 10
nums1 = np.array([128, 200, 10], dtype=np.uint8)
print(nums1)

# Масив 2: 250 10 34
nums2 = np.array([250, 10, 34], dtype=np.uint8)
print(nums2)

# Об’єднайте їх у пропорції 20% першого масив + 80%
# другого масиву. В результаті має бути тип даних uint8 та
# числа в діапазоні 0-255

nums1 = nums1.astype(np.float32)
nums2 = nums2.astype(np.float32)

sum_nums = (nums1 * 0.2) + (nums2 * 0.8)
mask = sum_nums > 255
sum_nums[mask] = 255
sum_nums = sum_nums.astype(np.uint8)
print(sum_nums)

