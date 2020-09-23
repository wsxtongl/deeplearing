import os ,shutil

#创建多级目录
#os.makedirs("file")
train_path0 = r"catdog_data/train/0"
test_path0 = r"catdog_data/test/0"
train_path1 = r"catdog_data/train/1"
test_path1 = r"catdog_data/test/1"

path = r'D:\BaiduNetdiskDownload\catdog\cat_dog\img'
filenames = os.listdir(path)
# for i in range(10):
#     os.mkdir(r"catdog_data/"+f'{i}')
#

# filenames.sort(key=lambda x: int(x[0]))
# print(filenames)

for i in range(len(filenames)):
    if i < 600:
        shutil.copy(path+'/'+filenames[i],test_path0)
    elif 600 <= i < 6000:
        shutil.copy(path + '/' + filenames[i], train_path0)
    elif 6000 <= i < 6600:
        shutil.copy(path + '/' + filenames[i], test_path1)
    elif 6600 <= i < 12000:
        shutil.copy(path + '/' + filenames[i], train_path1)
    else:
        break
    exit()


