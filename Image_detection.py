import os
from PIL import Image
import numpy as np

bg_path = r"D:\BaiduNetdiskDownload\20200723_深度学习05\test_20200723_02\data"
x = 1

for filename in os.listdir(bg_path):
    background = Image.open("{0}/{1}".format(bg_path,filename))
    bg_shape = np.shape(background)
    if len(bg_shape) == 3:
        background = background
    else:
        continue
    if background.mode == "P":
        background = background.convert('RGB')
    background = background.resize((300,300))
    fg_name = np.random.randint(1,21)
    fg_img = Image.open("./yellow/{0}.png".format(fg_name))
    limit_size = np.random.randint(50,180)
    fg_img = fg_img.resize((limit_size,limit_size))
    x1 = np.random.randint(0,300-limit_size)
    y1 = np.random.randint(0,300-limit_size)
    r,g,b,a = fg_img.split()
    background.paste(fg_img,(x1,y1),mask = a)
    x2 = x1 + limit_size
    y2 = y1 + limit_size
    background.save("./scenery/{0}{1}.jpg".format(x,"." + str(x1) + "." + str(y1) + "." + str(x2) + "."
                    + str(y2)))
    x+=1