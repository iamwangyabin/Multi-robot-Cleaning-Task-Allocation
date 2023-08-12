import os
import math
import cv2
import numpy as np
from Map import Array2D
import matplotlib.pyplot as plt

class AStar:
    """
    AStar算法的Python3.x实现
    """
    def __init__(self, map2d, startPoint, endPoint, BlockTag=0):
        """
        构造AStar算法的启动条件
        :param map2d: Array2D类型的寻路数组
        :param startPoint: Point或二元组类型的寻路起点
        :param endPoint: Point或二元组类型的寻路终点
        :param BlockTag: int类型的不可行走标记（若地图数据!=passTag即为障碍）
        """
        self.map2d = np.array(map2d.data)
        self.hight = self.map2d.shape[1]  # 行数->y
        self.width = self.map2d.shape[0]  # 列数->x
        self.BlockTag = BlockTag
        self.startPoint = startPoint
        self.endPoint = endPoint
        self.star = {'位置': (startPoint.x, startPoint.y), '代价': 700, '父节点': (startPoint.x, startPoint.y)}  # 起点
        self.end = {'位置': (endPoint.x, endPoint.y), '代价': 0, '父节点': (endPoint.x, endPoint.y)}  # 终点

    def run(self):
        openlist = []  # open列表，存储可能路径
        closelist = [self.star]  # close列表，已走过路径
        step_size = 2  # 搜索步长。
        # 步长太小，搜索速度就太慢。步长太大，可能直接跳过障碍，得到错误的路径
        # 步长大小要大于图像中最小障碍物宽度
        while 1:
            s_point = closelist[-1]['位置']  # 获取close列表最后一个点位置，S点
            add = ([0, step_size], [0, -step_size], [step_size, 0], [-step_size, 0])
                   # [step_size,step_size], [step_size, -step_size], [-step_size,step_size], [-step_size, -step_size])  # 可能运动的四个方向增量
            for i in range(len(add)):
                x = s_point[0] + add[i][0]  # 检索超出图像大小范围则跳过
                if x < 0 or x >= self.width:
                    continue
                y = s_point[1] + add[i][1]
                if y < 0 or y >= self.hight:  # 检索超出图像大小范围则跳过
                    continue
                G = abs(x - self.star['位置'][0]) + abs(y - self.star['位置'][1])  # 计算代价
                H = abs(x - self.end['位置'][0]) + abs(y - self.end['位置'][1])  # 计算代价
                F = G + H
                if H < 20:  # 当逐渐靠近终点时，搜索的步长变小
                    step_size = 1
                addpoint = {'位置': (x, y), '代价': F, '父节点': s_point}  # 更新位置
                count = 0
                for i in openlist:
                    if i['位置'] == addpoint['位置']:
                        count += 1
                for i in closelist:
                    if i['位置'] == addpoint['位置']:
                        count += 1
                if count == 0:  # 新增点不在open和close列表中
                    if self.map2d[x, y] != self.BlockTag:  # 非障碍物
                        openlist.append(addpoint)
            t_point = {'位置': (self.startPoint.x, self.startPoint.y), '代价': 10000, '父节点': (self.startPoint.x, self.startPoint.y)}
            for j in range(len(openlist)):  # 寻找代价最小点
                if openlist[j]['代价'] < t_point['代价']:
                    t_point = openlist[j]
            for j in range(len(openlist)):  # 在open列表中删除t点
                if t_point == openlist[j]:
                    openlist.pop(j)
                    break
            closelist.append(t_point)  # 在close列表中加入t点
            # cv2.circle(informap,t_point['位置'],1,(200,0,0),-1)
            if t_point['位置'] == self.end['位置']:  # 找到终点！！
                # print("找到终点")
                break
        # 逆向搜索找到路径
        road = []
        road.append(closelist[-1])
        point = road[-1]
        k = 0
        while 1:
            for i in closelist:
                if i['位置'] == point['父节点']:  # 找到父节点
                    point = i
                    # print(point)
                    road.append(point)
            if point == self.star:
                # print("路径搜索完成")
                break
        roadlength = 0
        for i in road:
            preX, preY = i["父节点"]
            X, Y = i["位置"]
            roadlength+=math.fabs(X - preX)+math.fabs(Y - preY)
        return road, roadlength



if __name__ == '__main__':
    fold_path = './data/set1/'
    images_name = os.listdir(fold_path)
    image_num = []
    for i in images_name:
        if len(i.split('_')) == 1:
            image_num.append(int(i.split('.')[0]))
    image_num = image_num[0]

    def connected_component_label(path):
        img = cv2.imread(path, 0)
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        num_labels, labels = cv2.connectedComponents(img)
        return num_labels, labels

    num_labels, labels = connected_component_label(fold_path + str(image_num) + '_rooms.png')
    centroid = {}
    depotposition = {"x": 0, "y": 0}
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
    wall = cv2.imread(fold_path + str(image_num) + '_wall.png', 0)
    multi = cv2.imread(fold_path + str(image_num) + '_multi.png', 0)
    map2d=Array2D(labels.shape[0],labels.shape[1])
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if multi[i][j] == 0:
                map2d[i][j] = 1
            if wall[i][j]!=0:
                map2d[i][j]=1

    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    startPoint = Point(int(centroid[1]["x"]), int(centroid[1]["y"]))
    endPoint = Point(int(centroid[3]["x"]), int(centroid[3]["y"]))

    starAL = AStar(map2d,startPoint,endPoint,1)
    root = starAL.run()
    #
    # import math
    # for i in root:
    #     preX, preY = i["父节点"]
    #     X,Y = i["位置"]
    #     if math.fabs(X-preX)>1:
    #
    #     map2d[i["位置"][0]][i["位置"][1]] = 1
    #
    # plt.imshow(np.array(map2d.data))
    # plt.show()
    #
