import os
import math
import time
import cmath
import random
import numpy as np
# import gurobipy as gp
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataloader import get_data, processSet1, processSet2
from Gurobi import GurobiSolver
from GA import GA as GASolver
from SA import SA as SASolver
from PSO import Model as PSOSolver
plt.rcParams['font.sans-serif']=['Times New Roman']

fold_path = './data/set2/'
depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
map2d, distance, centroid = get_data(fold_path = fold_path, imgpath=str(31868853), cal=True)

NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B = processSet2(fold_path, str(31868853), cal=False)

Solver = GurobiSolver(NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B)
# Solver = SASolver(NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B)
result = Solver.run()


# image = plt.imread(os.path.join(fold_path, '31868853.jpg'))
# image = np.array(image)
# drawroute(image, 0, 3, color = [255,0,0])
# drawroute(image, 3, 5, color = [255,0,0])
# drawroute(image, 5, 7, color = [255,0,0])
# drawroute(image, 7, 8, color = [255,0,0])
# drawroute(image, 8, 0, color = [255,0,0])
#
# drawroute(image, 0, 1, color = [0,255,0])
# drawroute(image, 1, 2, color = [0,255,0])
# drawroute(image, 2, 6, color = [0,255,0])
# drawroute(image, 6, 4, color = [0,255,0])
# drawroute(image, 4, 0, color = [0,255,0])
#
# drawroute(image, 0, 2, color = [0,0,255])
# drawroute(image, 2, 8, color = [0,0,255])
# drawroute(image, 8, 0, color = [0,0,255])
#
# drawroute(image, 0, 3, color = [0,255,255])
# drawroute(image, 3, 1, color = [0,255,255])
# drawroute(image, 1, 5, color = [0,255,255])
# drawroute(image, 5, 7, color = [0,255,255])
# drawroute(image, 7, 6, color = [0,255,255])
# drawroute(image, 6, 4, color = [0,255,255])
# drawroute(image, 4, 0, color = [0,255,255])
#
# plt.imshow(image)  # 显示图片
# plt.axis('off')  # 不显示坐标轴
# plt.savefig("./new_result2.png", dpi=200, bbox_inches='tight')


# def drawroute(image, startpoint, endpoint, color = [255,0,0]):
#     for i in distance[(startpoint,endpoint)][1]:
#         current = i['位置']
#         image[int(current[0])][int(current[1])] = color
#         pre = i['父节点']
#         image[int(pre[0])][int(pre[1])] = color
#         for ii in range(min(int(current[0]), int(pre[0])), max(int(current[0]), int(pre[0]))):
#             image[ii][int(current[1])] = color
#         for jj in range(min(int(current[1]), int(pre[1])), max(int(current[1]), int(pre[1]))):
#             image[int(current[0])][jj] = color


gasolver = GASolver(int(len(NJobs) / 2), {"R1": 0, "R2": 0, "R3": 1, "R4": 1}, RB=[2, 2], jobTypeNum=2, Service=D,
            Traveltime=traveltime, Agents=Agents)
garesult = gasolver.run()

sasolver = SASolver(int(len(NJobs) / 2), {"R1": 0, "R2": 0, "R3": 1, "R4": 1}, RB=[2, 2], jobTypeNum=2, Service=D,
            Traveltime=traveltime, Agents=Agents)
saresult = sasolver.mainrun()

psosolver = PSOSolver(int(len(NJobs) / 2), {"R1": 0, "R2": 0, "R3": 1, "R4": 1}, RB=[2, 2], jobTypeNum=2, Service=D,
               Traveltime=traveltime, Agents=Agents)
psosolver.setPara(epochs=500, popsize=2000, Vmax=5, w=0.8, c1=2, c2=2)
psoresult = psosolver.run()

ga_tour = []
RB = {"R1": 0, "R2": 0, "R3": 1, "R4": 1}
for k, v in RB.items():
    pre = 'depot'
    temptour = []
    for i,j,_ in gasolver.decode(garesult[0]):
        if j == k:
            temptour.append((pre, "room"+str(i)+"_"+str(RB[j]), k))
            pre = "room"+str(i)+"_"+str(RB[j])
    ga_tour.append(temptour)




# image = plt.imread(os.path.join(fold_path, '31868853.jpg'))
# image = np.array(image)
# drawroute(image, 0, 5, color = [255,0,0])
# drawroute(image, 5, 3, color = [255,0,0])
# drawroute(image, 3, 8, color = [255,0,0])
# drawroute(image, 8, 4, color = [255,0,0])
# drawroute(image, 4, 0, color = [255,0,0])
#
# drawroute(image, 0, 1, color = [0,255,0])
# drawroute(image, 1, 2, color = [0,255,0])
# drawroute(image, 2, 7, color = [0,255,0])
# drawroute(image, 7, 6, color = [0,255,0])
# drawroute(image, 6, 0, color = [0,255,0])
#
# drawroute(image, 0, 3, color = [0,0,255])
# drawroute(image, 3, 0, color = [0,0,255])
#
# drawroute(image, 0, 5, color = [0,255,255])
# drawroute(image, 5, 1, color = [0,255,255])
# drawroute(image, 1, 2, color = [0,255,255])
# drawroute(image, 2, 8, color = [0,255,255])
# drawroute(image, 8, 7, color = [0,255,255])
# drawroute(image, 7, 4, color = [0,255,255])
# drawroute(image, 4, 6, color = [0,255,255])
# drawroute(image, 6, 0, color = [0,255,255])
#
# plt.imshow(image)  # 显示图片
# plt.axis('off')  # 不显示坐标轴
# plt.savefig("./ga_result2.png", dpi=200, bbox_inches='tight')

sa_tour = []
RB = {"R1": 0, "R2": 0, "R3": 1, "R4": 1}
for k, v in RB.items():
    pre = 'depot'
    temptour = []
    for i,j,_ in sasolver.decode(saresult[0]):
        if j == k:
            temptour.append((pre, "room"+str(i)+"_"+str(RB[j]), k))
            pre = "room"+str(i)+"_"+str(RB[j])
    sa_tour.append(temptour)
#
#
# image = plt.imread(os.path.join(fold_path, '31868853.jpg'))
# image = np.array(image)
# drawroute(image, 0, 5, color = [255,0,0])
# drawroute(image, 5, 7, color = [255,0,0])
# drawroute(image, 7, 6, color = [255,0,0])
# drawroute(image, 6, 8, color = [255,0,0])
# drawroute(image, 8, 4, color = [255,0,0])
# drawroute(image, 4, 0, color = [255,0,0])
#
# drawroute(image, 0, 1, color = [0,255,0])
# drawroute(image, 1, 2, color = [0,255,0])
# drawroute(image, 2, 3, color = [0,255,0])
# drawroute(image, 3, 0, color = [0,255,0])
#
# drawroute(image, 0, 1, color = [0,0,255])
# drawroute(image, 1, 6, color = [0,0,255])
# drawroute(image, 6, 8, color = [0,0,255])
# drawroute(image, 8, 0, color = [0,0,255])
#
# drawroute(image, 0, 7, color = [0,255,255])
# drawroute(image, 7, 5, color = [0,255,255])
# drawroute(image, 5, 2, color = [0,255,255])
# drawroute(image, 2, 3, color = [0,255,255])
# drawroute(image, 3, 4, color = [0,255,255])
# drawroute(image, 4, 0, color = [0,255,255])
#
# plt.imshow(image)  # 显示图片
# plt.axis('off')  # 不显示坐标轴
# plt.savefig("./sa_result2.png", dpi=200, bbox_inches='tight')


pso_tour = []
RB = {"R1": 0, "R2": 0, "R3": 1, "R4": 1}
for k, v in RB.items():
    pre = 'depot'
    temptour = []
    for i,j,_ in psosolver.decode(psoresult[0].nodes_seq):
        if j == k:
            temptour.append((pre, "room"+str(i)+"_"+str(RB[j]), k))
            pre = "room"+str(i)+"_"+str(RB[j])
    pso_tour.append(temptour)

#
#
# image = plt.imread(os.path.join(fold_path, '31868853.jpg'))
# image = np.array(image)
# drawroute(image, 0, 5, color = [255,0,0])
# drawroute(image, 5, 7, color = [255,0,0])
# drawroute(image, 7, 8, color = [255,0,0])
# drawroute(image, 8, 6, color = [255,0,0])
# drawroute(image, 6, 0, color = [255,0,0])
#
# drawroute(image, 0, 1, color = [0,255,0])
# drawroute(image, 1, 2, color = [0,255,0])
# drawroute(image, 2, 3, color = [0,255,0])
# drawroute(image, 3, 4, color = [0,255,0])
# drawroute(image, 4, 0, color = [0,255,0])
#
# drawroute(image, 0, 2, color = [0,0,255])
# drawroute(image, 2, 0, color = [0,0,255])
#
# drawroute(image, 0, 1, color = [0,255,255])
# drawroute(image, 1, 7, color = [0,255,255])
# drawroute(image, 7, 8, color = [0,255,255])
# drawroute(image, 8, 5, color = [0,255,255])
# drawroute(image, 5, 3, color = [0,255,255])
# drawroute(image, 3, 4, color = [0,255,255])
# drawroute(image, 4, 6, color = [0,255,255])
# drawroute(image, 6, 0, color = [0,255,255])
#
# plt.imshow(image)  # 显示图片
# plt.axis('off')  # 不显示坐标轴
# plt.savefig("./pso_result2.png", dpi=200, bbox_inches='tight')


def plotGant(tours, totaltime, pdfname="sa.pdf"):
    starttime = {"depot":0.0}
    endtime = {"depot":0.0}
    for tour in tours[:2]:
        endtime["depot"] = 0
        starttime["depot"] = 0
        for i, j, k in tour:
            starttime[j] = traveltime[i, j] + endtime[i]
            endtime[j] = starttime[j]+D[j,k]
    for tour in tours[2:]:
        endtime["depot"] = 0
        starttime["depot"] = 0
        for i, j, k in tour:
            if j != 'depot':
                prejob = j.split('_')[0]+'_0'
                if endtime[prejob] > traveltime[i, j] + endtime[i]:
                    starttime[j] = endtime[prejob]
                else:
                    starttime[j] = traveltime[i, j] + endtime[i]
                endtime[j] = starttime[j]+D[j,k]
    color = ['royalblue', 'darkorange']
    labels = ['Vocuuming', 'Mopping']  # legend标签列表，上面的color即是颜色列表
    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]
    plt.xlabel('Time (s)')
    plt.ylabel('Cleaning Zone')
    for i in range(int(len(NJobs)/2)):
        num = i+1
        # plt.plot([0,int(result[0])], [num, num], color='grey', linewidth=1)
        name = "room"+str(num)+"_0"
        # plt.plot([0,starttime[name]], [num, num], color='grey', linewidth=1)
        plt.barh(num, endtime[name]-starttime[name], left=starttime[name], color=['royalblue'])
        # plt.plot([endtime[name],starttime["room"+str(num)+"_1"]], [num, num], color='grey', linewidth=1)
        name = "room"+str(num)+"_1"
        plt.barh(num, endtime[name]-starttime[name], left=starttime[name], color=['darkorange'])
        # plt.plot([endtime[name],totaltime], [num, num], color='grey', linewidth=1)



    plt.legend(handles=patches, loc='upper left', framealpha=1)  # 生成legend
    plt.savefig(pdfname)



import pdb; pdb.set_trace()


# plotGant(result[2], result[0], pdfname="guro.pdf")



plotGant(sa_tour, saresult[1], pdfname="sa.png")
# plotGant(ga_tour, garesult[1], pdfname="ga.pdf")
# plotGant(pso_tour, psoresult[1], pdfname="pso.pdf")
