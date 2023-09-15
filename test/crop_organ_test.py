# import numpy as np

# # 假设这是你的 target_region 数组
# target_region = np.array([[[0, 1, 1, 0],
#                             [0, 1, 2, 0],
#                             [0, 0, 2, 0]],
#                             [[0, 1, 1, 0],
#                             [0, 1, 2, 0],
#                             [0, 0, 2, 0]]])

# # 找到非零元素的索引
# nonzero_indices = np.where(np.isin(target_region, [1,2]))
# # nonzero_indices = np.isin(target_region, [1,2])
# print(nonzero_indices)
# # 找到最小和最大的行、列索引
# min_row = np.min(nonzero_indices[0])
# max_row = np.max(nonzero_indices[0])
# min_col = np.min(nonzero_indices[1])
# max_col = np.max(nonzero_indices[1])
# max_slice = np.max(nonzero_indices[2])
# min_slice = np.min(nonzero_indices[2])

# # 裁剪出边界上的非零区域
# cropped_target_region = target_region[min_row:max_row+1, min_col:max_col+1, min_slice:max_slice+1]

# print(cropped_target_region)




# import numpy as np

# # 自定义的运算函数
# def custom_operation(x):
#     return x**2 + 3

# # 创建一个NumPy数组
# array = np.array([[120, 150, 130],
#                   [125, 140, 135]])
# array = array.astype(np.float32)

# array2 = np.array([[0 , 29, 0],
#                   [25, 0, 0]])
# array2 = array2.astype(np.float32)

# array = array * - (array2 / 30 - 1) + array2 / 30 * 35

# array = array.astype(np.int16)
# # 将函数应用于数组的每个元素

# print(array)



# import numpy as np

# # 创建一个NumPy数组
# array = np.array([1, 2, 3, 4, 5])

# # 指定值列表
# values_to_check = [2, 4]

# # 使用np.isin函数获取元素是否在值列表中的布尔数组
# result = np.isin(array, values_to_check)

# # 获取np.isin函数结果的相反结果
# inverse_result = ~result

# print("原始数组:")
# print(array)
# print("np.isin函数结果:")
# print(result)
# print("np.isin函数结果的相反结果:")
# print(inverse_result)

# import cv2
# import numpy as np

# # 创建一个3维的随机二值图像作为示例输入
# depth = 5
# height, width = 256, 256
# input_image = np.random.randint(0, 2, size=(depth, height, width), dtype=np.uint8)

# # 定义开运算的核（结构元素）
# kernel_size = (5, 5, 5)
# kernel = np.ones(kernel_size, dtype=np.uint8)

# # 进行开运算
# output_image = cv2.morphologyEx(input_image, cv2.MORPH_OPEN, kernel)

# # 打印输入和输出图像的形状
# print("输入图像形状:", input_image.shape)
# print("输出图像形状:", output_image.shape)


# import numpy as np
# from skimage.measure import label, regionprops

# # 创建一个示例的3维二值数组
# binary_array = np.array([[[0, 1, 1],
#                           [1, 0, 0],
#                           [0, 1, 1]],

#                          [[1, 0, 1],
#                           [0, 1, 0],
#                           [1, 0, 0]],

#                          [[0, 0, 0],
#                           [1, 1, 1],
#                           [0, 0, 0]]])

# # 定义保留的联通域的最小尺寸
# k = 10

# # 进行联通域分析
# labeled_array, num_features = label(binary_array, connectivity=3, return_num=True)

# # 获取联通域的属性
# regions = regionprops(labeled_array)

# # 创建一个新的数组，初始化为0
# new_array = np.zeros_like(binary_array)

# # 遍历每个联通域，保留大小大于k的联通域
# for region in regions:
#     if region.area > k:
#         label_value = region.label
#         new_array[labeled_array == label_value] = 1

# print("保留的联通域的新数组:")
# print(new_array)


# import numpy as np

# # 创建一个示例数组
# array = np.array([[0, 0, 3, 0],
#                   [0, 5, 0, 0],
#                   [0, 0, 0, 8]])

# # 获取每个维度上第一个不为零的索引
# nonzero_indices = [np.where(array.any(axis=i)) for i in range(array.ndim)]

# print("Non-zero indices for each dimension:", nonzero_indices)
# print(array[0:,1:])

# import numpy as np

# random = np.random.uniform(1,5)

# print(random)

# import SimpleITK as sitk

# # Load the image
# img_path = '/ccvl/net/ccvl15/zzhou82/PublicAbdominalData/04_LiTS/img/liver_0.nii.gz'
# mask_path = '/ccvl/net/ccvl15/zzhou82/PublicAbdominalData/04_LiTS/label/liver_0.nii.gz'
# img = sitk.ReadImage(img_path)
# img = sitk.GetArrayFromImage(img)

# mask = sitk.ReadImage(mask_path)
# mask = sitk.GetArrayFromImage(mask)

# print(img.shape)
# save = sitk.GetImageFromArray(img)
# sitk.WriteImage(save, './data/img/liver_0.nii.gz')
# save = sitk.GetImageFromArray(mask)
# sitk.WriteImage(save, './data/mask/liver_0.nii.gz')

import numpy as np

# 创建一个示例数组
array = np.array([[0, 0, 3, 0],
                    [0, 5, 0, 0],
                    [0, 0, 0, 8]])

array2 = (array * array) >= 1

print(array+array2)