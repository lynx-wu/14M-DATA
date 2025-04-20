from PIL import Image
import os

file_dir = "/media/disk/01drive/06chengyang/14M_origin/jieya/Fundus/"            # 原始图片路径
rotate_dir = "/media/disk/01drive/06chengyang/14M/02"    # 保存路径
count = 0

for img_name in os.listdir(file_dir):
    count += 1
    if img_name != "1258785.jpeg":
        continue
    img_path = file_dir + img_name     #批量读取图片
    img = Image.open(img_path)
    im_rotate = img.rotate(180)           # 指定逆时针旋转的角度
    # im_rotate = img.transpose(Image.FLIP_LEFT_RIGHT)
    save_name = img_name.split('/')[0].split('.')[0] + "_2.jpg"
    # print(rotate_dir + "/" + save_name)
    print(str(100*count/1371861)+"%  "+str(count))
    im_rotate.save(rotate_dir + "/" + save_name)  # 保存旋转后的图片

