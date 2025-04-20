from PIL import Image
import os
import numpy as np
import random

# 定义源文件夹和目标文件夹
source_folder = "/media/disk/01drive/06chengyang/14M_origin/jieya/Fundus/"  # 原始图片地址
target_folder = "/media/disk/01drive/06chengyang/14M/07"  # 添加高斯噪声后的图片存放地址
count = 0
# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 获取源文件夹中的所有文件名
file_names = os.listdir(source_folder)

# 遍历每个文件
for file_name in file_names:
    # 构建完整的文件路径
    file_path = os.path.join(source_folder, file_name)
    count += 1
    # 确保是图片文件
    if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 打开图片
        img = Image.open(file_path)
        
        # 将图片转换为NumPy数组
        img_array = np.array(img)
        
        # 检查图像的通道数
        if img_array.ndim == 2:  # 灰度图
            rows, cols = img_array.shape
            channels = 1
            sigma = random.uniform(0, 20)  # 标准差范围20,可以根据需要调整
            # 生成高斯噪声
            gauss = np.random.normal(0, sigma, (rows, cols))
            # 将高斯噪声添加到图像上
            noisy_img_array = img_array + gauss
            # 限制像素值在0到255之间
            noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
        else:  # 彩色图
            rows, cols, channels = img_array.shape
            sigma = random.uniform(0, 20)  # 标准差范围20,可以根据需要调整
            # 生成高斯噪声
            gauss = np.random.normal(0, sigma, (rows, cols, channels))
            # 将高斯噪声添加到图像上
            noisy_img_array = img_array + gauss
            # 限制像素值在0到255之间
            noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)

        noisy_img = Image.fromarray(noisy_img_array)
        # 生成新的文件名，添加噪声标准差作为后缀
        new_file_name = f"{os.path.splitext(file_name)[0]}_7.jpg"
        target_file_path = os.path.join(target_folder, new_file_name)
        print(str(100*count/1371861)+"%  "+str(count))
        # 保存添加高斯噪声后的图片
        noisy_img.save(target_file_path)
