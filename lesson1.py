import numpy as np

nums = np.array([1, 2, 3, 4]) # массив
print(nums)
print(type(nums))

print(nums.shape)
print(nums.dtype)


# двумерный массив
nums = np.array(
    [[1, 2, 3, 4],
    [5, 6, 7, 8],
     [9, 10, 11, 12]])

print(nums)

# Создание массивов
# со списка
nums_list = [1, 2, 3, 4]
nums = np.array(nums_list)

# аналог range
nums = np.arange(10,20)
print(nums)

# массив нулей, 1 или случайных
nums = np.zeros((6,))
print(nums)

nums = np.ones((3,3))
print(nums)

nums = np.random.rand(2, 3)
print(nums)

# изменение размеров и типов

nums = np.arange(12)
print(nums)
print(nums.shape)
print(nums.dtype)

new_nums = nums.reshape(3,4)
print(new_nums)
print(new_nums.shape)
print(new_nums.dtype)

nums_float16 = nums.astype(np.float16)
print(nums_float16)
print(nums_float16.shape)
print(nums_float16.dtype)


# перенаполнение
nums = np.array([10, 20, 30], dtype=np.int8) # int8 - это цифры от -128 до 127
print(nums)


# индексация
nums = np.array([10, 20, 30, 40, 50])
print(nums[0]) # - 1 элемент
print(nums[-1]) # - последний элемент
print(nums[1:3]) # - 20 - 40


# индексация таблиц
nums = np.arange(12).reshape(4,3)
print(nums)

# сперва рядки потом столбцы
print(nums[1,2])
print(nums[3]) # - весь ряд
print(nums[0:2]) # - 2 первых ряда


# булевые маски
nums = np.array([15, 8, 12, 4, 1, 2])
mask = nums > 10
print(mask)

# все элементы что отвечают маске
print(nums[mask])

