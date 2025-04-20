from PIL import Image, ImageEnhance
import os
import random

# 定义源文件夹和目标文件夹
source_folder = "/media/disk/01drive/06chengyang/14M_origin/jieya/Fundus/"  # 原始图片地址
target_folder = "/media/disk/01drive/06chengyang/14M/06"  # 调整饱和度后的图片存放地址
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
    
    # 确保是图片文件
    if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 打开图片
        img = Image.open(file_path)
        
        count += 1 
        # 随机生成饱和度调整因子
        saturation_factor = random.uniform(0.8, 1.2)
        
        # 创建一个饱和度增强器
        enhancer = ImageEnhance.Color(img)
        
        # 调整饱和度
        img_enhanced = enhancer.enhance(saturation_factor)
        
        # 生成新的文件名，添加饱和度调整因子作为后缀
        new_file_name = f"{os.path.splitext(file_name)[0]}_6.jpg"
        
        # 构建目标文件路径
        target_file_path = os.path.join(target_folder, new_file_name)
        print(str(100*count/1371861)+"%  "+str(count))
        # 保存调整饱和度后的图片
        img_enhanced.save(target_file_path)
