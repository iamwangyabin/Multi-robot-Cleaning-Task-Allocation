import os
import cv2
import numpy as np
from Map import Array2D

import matplotlib.path as mpltPath
import matplotlib.pyplot as plt

fold_path = './data/set5/'
images_name = os.listdir(fold_path)
image_num = []
for i in images_name:
    if len(i.split('_')) == 1:
        image_num.append(int(i.split('.')[0]))

def connected_component_label(path):
    img = cv2.imread(path, 0)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels = cv2.connectedComponents(img)
    return num_labels, labels

fontFace = cv2.FONT_HERSHEY_COMPLEX
fontScale = 0.7
fontcolor = (0, 0, 255) # BGR
thickness = 1
lineType = 4
bottomLeftOrigin = 1

depot = {}
for nimg in image_num:
    while(1):
        print(nimg)
        num_labels, labels = connected_component_label(fold_path + str(nimg) + '_rooms.png')
        centroid = {}
        for i in range(1, num_labels):
            centroid[i] = {"x": 0, "y": 0, "num": 0}
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] != 0:
                    centroid[labels[i, j]]["x"] = centroid[labels[i, j]]["x"] + i
                    centroid[labels[i, j]]["y"] = centroid[labels[i, j]]["y"] + j
                    centroid[labels[i, j]]["num"] = centroid[labels[i, j]]["num"] + 1
        for i in range(1, num_labels):
            centroid[i]["x"] = centroid[i]["x"] / centroid[i]["num"]
            centroid[i]["y"] = centroid[i]["y"] / centroid[i]["num"]

        img = cv2.imread(fold_path + str(nimg) + '.jpg')
        img_seg = cv2.imread(fold_path + str(nimg) + '_multi.png')

        wall = cv2.imread(fold_path + str(nimg) + '_wall.png', 0)
        multi = cv2.imread(fold_path + str(nimg) + '_multi.png', 0)
        map2d = Array2D(labels.shape[0], labels.shape[1])

        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if multi[i][j] == 0:
                    map2d[i][j] = 1
                if wall[i][j] != 0:
                    map2d[i][j] = 1

        plt.imshow(np.array(map2d.data))  # 显示图片
        plt.axis('off')  # 不显示坐标轴
        plt.show()

        def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                depot[nimg] = [x,y]
                print("x,y:{},{}".format(str(x),str(y)))
        cv2.namedWindow("image")
        cv2.namedWindow("seg")
        cv2.setMouseCallback("seg", on_EVENT_LBUTTONDOWN)
        for i in range(1, num_labels):
            cv2.circle(img_seg, (int(centroid[i]["y"]), int(centroid[i]["x"])), 1, (0, 0, 255), 4)
            cv2.putText(img, str(centroid[i]["num"]), (int(centroid[i]["y"]), int(centroid[i]["x"])),fontFace, fontScale, fontcolor, thickness, lineType)
            # cv2.te(img_seg, (int(centroid[i]["y"]), int(centroid[i]["x"])), 1, (0, 0, 255), 4)
        cv2.imshow("image", img)
        cv2.imshow("seg", img_seg)
        if cv2.waitKey(0)&0xFF==27:
            break

cv2.destroyAllWindows()
np.save('set5.npy', depot)


# 为了方便筛选数据集制作，返回每张图的房间数目
fold_path = './data/test/'
numRoomsList = {}
for i in image_num:
    num_labels, labels = connected_component_label(fold_path + str(i) + '_rooms.png')
    if num_labels not in numRoomsList.keys():
        numRoomsList[num_labels] = [i]
    else:
        numRoomsList[num_labels].append(i)


fold_path = './data/set1/'
images_name = os.listdir(fold_path)
image_num = []
for i in images_name:
    if len(i.split('_')) == 1:
        image_num.append(int(i.split('.')[0]))

fold_path = './data/set1/'
depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
len(depotDic.keys())

fold_path = './data/set3_add/'
images_name = os.listdir(fold_path)
image_num = []
for i in images_name:
    if len(i.split('_')) == 1:
        image_num.append(str(i.split('.')[0]))

def judge(c):
    num =0
    rith=0
    for i in image_num:
        if str(c) in i:
            rith=i
            num+=1
    if num ==1:
        return rith
    else:
        return False

import cmath
scale = np.load("scale.npy", allow_pickle=True).item()
def putin(rel, pix, cc):
    scale[cc] = cmath.sqrt(rel / pix).real
