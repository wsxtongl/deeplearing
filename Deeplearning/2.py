import os
from PIL import Image
import numpy as np
# path1 = r"./rename_scenery"
# path2 = r"./scenery"
# n2_list = []
# n1_list = []
# for filename2 in os.listdir(path2):
#     name2 = filename2.split(".")[1:5]
#     n2_list.append(name2)
#
# for i, file in enumerate(os.listdir(path1)):
#     NewName = os.path.join(path1, str(i+1) + '.' + str(n2_list[i][0]) + '.' + str(n2_list[i][1]) + '.'
#       + str(n2_list[i][2]) + '.' + str(n2_list[i][3]) + '.' + 'jpg')
#     OldName = os.path.join(path1, file)
#     os.rename(OldName, NewName)
def iou( rec1, rec2):
    s_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    s_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    s_sum = s_rec1 + s_rec2
    # 左上点都取最大
    w1 = max(rec1[0], rec2[0])
    h1 = max(rec1[1], rec2[1])
    # 右下角都取最小
    w2 = min(rec1[2], rec2[2])
    h2 = min(rec1[3], rec2[3])
    if w1 >=w2 or h1 >= h2:
        return 0
    w = w2 - w1
    h = h2 - h1
    # 相交矩形面积
    intersect = w * h
    if intersect <= 0 :
        return 0
    else:
        iou = intersect / (s_sum - intersect)
        return iou
# a1 = np.array([0.65,185,277,299,364])
# a2 = np.array([0.9,255,197,327,320])
# a3 = np.array([0.82,281,225,359,340])
# b1 = np.array([0.88,357,57,409,119])
# b2 = np.array([0.75,374,69,426,140])
# c1 = np.array([0.83,522,150,579,229])
# c2 = np.array([0.62,549,118,633,195])
data1 = np.array([[0.65,185,277,299,364],[0.9,255,197,327,320],[0.82,281,225,359,340],
          [0.88,357,57,409,119],[0.75,374,69,426,140],[0.83,522,150,579,229],
          [0.62,549,118,633,195]])
print(data1[:,0])
#获取值的从小到大索引
data = np.argsort(data1[:,0])
number = data[::-1]
nm_list = []
def Nms(number):
    number_list = []
    print("剩余矩形框个数：",len(number))
    if len(number) <= 0:
        return 0
    else:
        nm_list.append(data1[number[0]])
        for i in range(len(number)-1):
            a_list = list(data1[number[0], 1:])
            b_list = list(list(data1[number[i+1],1:]))
            iu = iou(a_list, b_list)
            print(iu)
            if iu == 0:
                number_list.append(number[i+1])
        return Nms(number_list)
Nms(number)
rect = np.array(nm_list)
print(rect)
