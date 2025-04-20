import os
import random
from PIL import ImageEnhance, Image
file_dir = "/media/disk/01drive/06chengyang/14M_origin/jieya/Fundus/"
file_dir_save = "/media/disk/01drive/06chengyang/14M/08"

count = 0

for filename in os.listdir(file_dir):
    img_path = os.path.join(file_dir, filename)
    count += 1

    with Image.open(img_path) as img:
        if img is not None:
            bright_random = random.uniform(0.8, 1.2)

            img_bright = ImageEnhance.Brightness(img).enhance(bright_random)
            save_path = os.path.join(file_dir_save, filename).split(".")
            save = save_path[0] + "_8" + "." + save_path[1]
            print(str(100*count/1371861)+"%  "+str(count))
            img_bright.save(save)




