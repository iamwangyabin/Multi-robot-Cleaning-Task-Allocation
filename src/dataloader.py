import os
import cv2
import sys
import cmath
import numpy as np
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
from collections import defaultdict
from Map import Array2D
from Astar import AStar

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def connected_component_label(path):
    img = cv2.imread(path, 0)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels = cv2.connectedComponents(img)
    return num_labels, labels

def get_data(fold_path = './data/set1/', imgpath=str(28025487), cal=True):
    depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
    depotX = depotDic[int(imgpath)][1]
    depotY = depotDic[int(imgpath)][0]
    num_labels, labels = connected_component_label(fold_path + imgpath + '_rooms.png')
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

    wall = cv2.imread(fold_path + imgpath + '_wall.png', 0)
    multi = cv2.imread(fold_path + imgpath + '_multi.png', 0)
    map2d=Array2D(labels.shape[0],labels.shape[1])

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if multi[i][j] == 0:
                map2d[i][j] = 1
            if wall[i][j]!=0:
                map2d[i][j]=1

    if cal:
        distance={}
        for i in range(1, num_labels):
            for j in range(i, num_labels):
                if i != j:
                    print(str(i)+":"+str(j))
                    aStar = AStar(map2d,Point(int(centroid[i]["x"]), int(centroid[i]["y"])),
                                  Point(int(centroid[j]["x"]), int(centroid[j]["y"])), BlockTag=1)
                    pathList, length = aStar.run()
                    distance[i,j] = [length, pathList]
                    distance[j,i] = [length, pathList]
                if i == j:
                    distance[i, j] = [0, 0]
        # depot代表0
        for i in range(1, num_labels):
            print(str(0) + ":" + str(i))
            aStar = AStar(map2d, Point(depotX, depotY),
                          Point(int(centroid[i]["x"]), int(centroid[i]["y"])), BlockTag=1)
            pathList, length = aStar.run()
            distance[i, 0] = [length, pathList]
            distance[0, i] = [length, pathList]

        np.save('{}_distance.npy'.format(imgpath), distance)
    else:
        distance = np.load('{}_distance.npy'.format(imgpath), allow_pickle=True).item()
    return map2d, distance, centroid


def processSet1(fold_path, path, cal=True):
    scaleDic = np.load("scale.npy", allow_pickle=True).item()
    scale = scaleDic[str(path)]
    map2d, distance, centroid = get_data(fold_path = fold_path, imgpath=path, cal=cal)
    AgentsAbility = {"R1": 0.016, "R3": 0.04}  # m2/s
    AgentSpeed = 0.2  # m/s
    Agents = ['R1', 'R3']  # workers
    numroom = len(centroid)
    RoomSize = {"depot": 0}
    Jobs0, Jobs1, NJobs = [], [], []
    Jobs = ["depot"]
    for i in range(1, numroom + 1):
        RoomSize["room" + str(i)] = centroid[i]['num'] * (scale * scale)
        Jobs0.append("room" + str(i) + "_0")
        Jobs1.append("room" + str(i) + "_1")
        Jobs.append("room" + str(i) + "_0")
        Jobs.append("room" + str(i) + "_1")
        NJobs.append("room" + str(i) + "_0")
        NJobs.append("room" + str(i) + "_1")
    P = defaultdict(lambda: 0)
    for i, j in zip(Jobs0, Jobs1):
        P[i, j] = 1
    B = defaultdict(lambda: 0)
    for i in Jobs1:
        B[i, "R3"] = 1
    for i in Jobs0:
        B[i, "R1"] = 1
    D = defaultdict(lambda: 0)
    for room in Jobs1:
        D[room, "R3"] = int(RoomSize[room[:5]] / AgentsAbility["R3"])
    for room in Jobs0:
        D[room, "R1"] = int(RoomSize[room[:5]] / AgentsAbility["R1"])
    traveltime = {}
    for i in range(len(Jobs0) + 1):
        for j in range(len(Jobs0) + 1):
            if i != 0 and j == 0:
                traveltime["room" + str(i) + "_0", "depot"] = int(distance[i, 0][0] * scale / AgentSpeed)
                traveltime["room" + str(i) + "_1", "depot"] = int(distance[i, 0][0] * scale / AgentSpeed)
            elif i != 0 and j != 0:
                traveltime["room" + str(i) + "_0", "room" + str(j) + "_0"] = int(distance[i, j][0] * scale / AgentSpeed)
                traveltime["room" + str(i) + "_1", "room" + str(j) + "_1"] = int(distance[i, j][0] * scale / AgentSpeed)
                traveltime["room" + str(i) + "_1", "room" + str(j) + "_0"] = int(distance[i, j][0] * scale / AgentSpeed)
                traveltime["room" + str(i) + "_0", "room" + str(j) + "_1"] = int(distance[i, j][0] * scale / AgentSpeed)
            elif i == 0 and j != 0:
                traveltime["depot", "room" + str(j) + "_0"] = int(distance[0, j][0] * scale / AgentSpeed)
                traveltime["depot", "room" + str(j) + "_1"] = int(distance[0, j][0] * scale / AgentSpeed)
            else:
                traveltime["depot", "depot"] = 0
    return NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B


def processSet2(fold_path, path, cal=True):
    scaleDic = np.load("scale.npy", allow_pickle=True).item()
    scale = scaleDic[str(path)]
    map2d, distance, centroid = get_data(fold_path = fold_path, imgpath=path, cal=cal)
    AgentsAbility = {"R1": 0.016, "R2": 0.023, "R3": 0.04, "R4": 0.07}  # m2/s
    AgentSpeed = 0.2  # m/s
    Agents = ["R1", "R2", "R3", "R4"]  # workers
    numroom = len(centroid)
    RoomSize = {"depot": 0}
    Jobs0, Jobs1, NJobs = [], [], []
    Jobs = ["depot"]
    for i in range(1, numroom + 1):
        RoomSize["room" + str(i)] = centroid[i]['num'] * (scale * scale)
        Jobs0.append("room" + str(i) + "_0")
        Jobs1.append("room" + str(i) + "_1")
        Jobs.append("room" + str(i) + "_0")
        Jobs.append("room" + str(i) + "_1")
        NJobs.append("room" + str(i) + "_0")
        NJobs.append("room" + str(i) + "_1")
    P = defaultdict(lambda: 0)
    for i, j in zip(Jobs0, Jobs1):
        P[i, j] = 1
    B = defaultdict(lambda: 0)
    for i in Jobs1:
        B[i, "R3"] = 1
        B[i, "R4"] = 1
    for i in Jobs0:
        B[i, "R1"] = 1
        B[i, "R2"] = 1
    D = defaultdict(lambda: 0)
    for room in Jobs1:
        D[room, "R3"] = int(RoomSize[room[:5]] / AgentsAbility["R3"])
        D[room, "R4"] = int(RoomSize[room[:5]] / AgentsAbility["R4"])
    for room in Jobs0:
        D[room, "R1"] = int(RoomSize[room[:5]] / AgentsAbility["R1"])
        D[room, "R2"] = int(RoomSize[room[:5]] / AgentsAbility["R2"])
    traveltime = {}
    for i in range(len(Jobs0) + 1):
        for j in range(len(Jobs0) + 1):
            if i != 0 and j == 0:
                traveltime["room" + str(i) + "_0", "depot"] = int(distance[i, 0][0] * scale / AgentSpeed)
                traveltime["room" + str(i) + "_1", "depot"] = int(distance[i, 0][0] * scale / AgentSpeed)
            elif i != 0 and j != 0:
                traveltime["room" + str(i) + "_0", "room" + str(j) + "_0"] = int(distance[i, j][0] * scale / AgentSpeed)
                traveltime["room" + str(i) + "_1", "room" + str(j) + "_1"] = int(distance[i, j][0] * scale / AgentSpeed)
                traveltime["room" + str(i) + "_1", "room" + str(j) + "_0"] = int(distance[i, j][0] * scale / AgentSpeed)
                traveltime["room" + str(i) + "_0", "room" + str(j) + "_1"] = int(distance[i, j][0] * scale / AgentSpeed)
            elif i == 0 and j != 0:
                traveltime["depot", "room" + str(j) + "_0"] = int(distance[0, j][0] * scale / AgentSpeed)
                traveltime["depot", "room" + str(j) + "_1"] = int(distance[0, j][0] * scale / AgentSpeed)
            else:
                traveltime["depot", "depot"] = 0
    return NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B















# fold_path = './data/set1/'
# imgpath = str(45613198)
# depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
# depotX = depotDic[int(imgpath)][1]
# depotY = depotDic[int(imgpath)][0]
# num_labels, labels = connected_component_label(fold_path + imgpath + '_rooms.png')
# centroid = {}
# for i in range(1, num_labels):
#     centroid[i] = {"x": 0, "y": 0, "num": 0}
# for i in range(labels.shape[0]):
#     for j in range(labels.shape[1]):
#         if labels[i, j] != 0:
#             centroid[labels[i, j]]["x"] = centroid[labels[i, j]]["x"] + i
#             centroid[labels[i, j]]["y"] = centroid[labels[i, j]]["y"] + j
#             centroid[labels[i, j]]["num"] = centroid[labels[i, j]]["num"] + 1
# for i in range(1, num_labels):
#     centroid[i]["x"] = centroid[i]["x"] / centroid[i]["num"]
#     centroid[i]["y"] = centroid[i]["y"] / centroid[i]["num"]
#
# wall = cv2.imread(fold_path + imgpath + '_wall.png', 0)
# multi = cv2.imread(fold_path + imgpath + '_multi.png', 0)
# map2d=Array2D(labels.shape[0],labels.shape[1])
#
# for i in range(labels.shape[0]):
#     for j in range(labels.shape[1]):
#         if multi[i][j] == 0:
#             map2d[i][j] = 1
#         if wall[i][j]!=0:
#             map2d[i][j]=1
#
#
#
# map2d, distance, centroid =     get_data(fold_path = './data/set1/', imgpath=str(28025487), cal=True)

# imgpath = str(image_num[5])
# map2d, distance, centroid = get_data(imgpath, cal=True, depot = 2)
# for i in range(7):
#     for j in range(7):
#         if i != j:
#             points = distance[i,j][1]
#             for p in points:
#                 map2d[p.x][p.y] = 1
#
# for key, value in distance:
#     if value[0]!=0:
#         for i in value[1]:
#             map2d[i.x][i.y] = 1
#
# for i in range(1, num_labels):
#     map2d[int(centroid[i]["x"])][int(centroid[i]["y"])] = 1
# map2d[depotDic[45613198][1]][depotDic[45613198][0]] = 1
# # #
# plt.imshow(np.array(map2d.data))  # 显示图片
# plt.axis('off')  # 不显示坐标轴
# plt.show()
# #
#
# points = distance[2,9][1]
# for p in points:
#     map2d[p.x][p.y] = 1