import os
import math
import time
import cmath
import random
import numpy as np
# import gurobipy as gp
from collections import defaultdict
import matplotlib.pyplot as plt
from dataloader import get_data, processSet1, processSet2

"""
退火算法中求解空间编码规则
【任务1编号，任务2编号，机器人1（长度为任务1长度），。。。】
例如：
【1，2，3，4，5，|1，2，3，4，5，|1，2，3，4，5】
每种任务有5个，但是为了能够让机器人能够分配，需要给机器人安排随机性，在机器人序列中排第一的是第一个该类型机器人完成第几个任务。
"""

class SA:
    def __init__(self, RoomNum, robot_fix, RB, jobTypeNum, Service, Traveltime, Agents):
        self.RoomNum = RoomNum
        self.robot_fix = robot_fix
        self.RB = RB
        self.jobTypeNum = jobTypeNum
        self.Service = Service
        self.Traveltime = Traveltime
        self.Agents = Agents
        self.T0 = 500 #2000   set1 10，1
        self.r = 0.997
        self.Ts = 1 #0.1
        self.Lk = 300
        self.maxIterate = 30000

    def randomGenerateT(self):
        T = []
        for i in range(self.jobTypeNum):
            tempjob = []
            for i in range(self.RoomNum):
                tempjob.append((i + 1))
            random.shuffle(tempjob)
            T.extend(tempjob)
        for j in self.RB:
            tempjob = []
            for i in range(self.RoomNum):
                tempjob.append((i + 1))
            random.shuffle(tempjob)
            T.extend(tempjob)
        return T

    def isFeasible(self, T):
        joblength = T[-self.RoomNum * len(self.RB):]
        flag = True
        pre = -1
        for i in range(len(self.RB)):  # len(RB)
            star = i * self.RoomNum
            for j in range(self.RB[i]):  # RB[i]
                if joblength[star + j] <= pre:
                    flag = False
                if j == self.RB[i]-1 and joblength[star + j] != self.RoomNum:
                    flag = False
        return flag

    def initSolution(self):
        while True:
            dd = self.randomGenerateT()
            if self.isFeasible(dd):
                break
        return dd

    def decode(self, T):
        """
        :param T: 就是一组数据，比如123456，函数就是要解析这个编码返回结果（用时之类的）
        :return: 给出来每个任务开始时间，谁来服务等，用个列表表示
        """
        T = T.copy()
        Joblist = {}  # 输出类似于 {'R1': [5, 4], 'R2': [3, 2], 'R3': [1, 6]}
        currentRobotName = 0
        jjNum = 0
        for jj in self.RB:
            pre = 0
            for ii in range(jj):
                Joblist[self.Agents[currentRobotName]] = T[jjNum * self.RoomNum + pre:jjNum * self.RoomNum + T[
                    -self.RoomNum * (len(self.RB) - jjNum) + ii]]
                pre = T[-self.RoomNum * (len(self.RB) - jjNum) + ii]
                currentRobotName += 1
            jjNum += 1
        # 任务是有顺序的
        # 首先把每个路径都提出来单独计算，如果有冲突二次调整，对于T 我们假定任务序列前面的比后面的要先完成，因此使用前面的值计算后面的即可
        Result = []
        prejobTime = defaultdict(lambda: 0)
        for (robot, jobs) in Joblist.items():
            if self.robot_fix[robot] == 0:
                temptime = 0
                pre = "depot"
                for j in jobs:
                    temptime = temptime + self.Traveltime[pre, "room" + str(j) + "_" + str(self.robot_fix[robot])] + self.Service[
                        "room" + str(j) + "_" + str(self.robot_fix[robot]), robot]
                    Result.append([j, robot, temptime])
                    prejobTime["room" + str(j)] = temptime
                    pre = "room" + str(j) + "_" + str(self.robot_fix[robot])
            if self.robot_fix[robot] == 1:
                temptime = 0
                pre = "depot"
                for j in jobs:
                    if temptime + self.Traveltime[pre, "room" + str(j) + "_" + str(self.robot_fix[robot])] > prejobTime["room" + str(j)]:
                        temptime = temptime + self.Traveltime[pre, "room" + str(j) + "_" + str(self.robot_fix[robot])] + self.Service[
                            "room" + str(j) + "_" + str(self.robot_fix[robot]), robot]
                    else:
                        temptime = temptime + prejobTime["room" + str(j)] + self.Service[
                            "room" + str(j) + "_" + str(self.robot_fix[robot]), robot]
                    Result.append([j, robot, temptime])
                    prejobTime["room" + str(j)] = temptime
                    pre = "room" + str(j) + "_" + str(self.robot_fix[robot])
        return Result

    def parameter(self, Result):
        """
        函数作用是解码，也就是接收Result然后返回损失函数值
        :param Result:
        :return: 返回 Cmax
        """
        pre = Result[0][1]
        pretime = Result[0][2]
        prejob = Result[0][0]
        Cmax = 0
        for i in Result:
            if i[1] == pre:
                pretime = i[2]
                prejob = i[0]
            else:
                if Cmax < pretime + self.Traveltime["room" + str(prejob) + "_0", "depot"]:
                    Cmax = pretime + self.Traveltime["room" + str(prejob) + "_0", "depot"]
                pre = i[1]
                pretime = i[2]
                prejob = i[0]
        if Cmax < pretime + self.Traveltime["room" + str(prejob) + "_0", "depot"]:
            Cmax = pretime + self.Traveltime["room" + str(prejob) + "_0", "depot"]
        return Cmax

    def createNeibor(self, T, mode):
        """
        T: 输入的路径，解
        mode：使用的生成模式
        """
        while True:
            if mode == 1:
                newT = self.Swap(T)
            elif mode == 2:
                newT = self.Reversion(T)
            else:
                newT = self.Swap(T)
            if self.isFeasible(newT):
                break
        return newT

    def Swap(self, T):
        SwapT = T.copy()
        for j in range(len(self.RB)):
            r = random.sample([i for i in range(self.RoomNum)], 2)
            i1 = T[r[0] + j * self.RoomNum]
            i2 = T[r[1] + j * self.RoomNum]
            SwapT[r[0] + j * self.RoomNum] = i2
            SwapT[r[1] + j * self.RoomNum] = i1
        for j in range(len(self.RB)):
            r = random.sample([i for i in range(self.RoomNum)], 2)
            i1 = T[r[0] - (len(self.RB) - j) * self.RoomNum]
            i2 = T[r[1] - (len(self.RB) - j) * self.RoomNum]
            SwapT[r[0] - (len(self.RB) - j) * self.RoomNum] = i2
            SwapT[r[1] - (len(self.RB) - j) * self.RoomNum] = i1
        return SwapT

    def Reversion(self, T):
        Rever = T.copy()
        for j in range(len(self.RB)):
            r = random.sample([i for i in range(self.RoomNum)], 2)
            i1 = min(r)
            i2 = max(r)
            Rever[i1 + j * self.RoomNum:i2 + j * self.RoomNum+1] = T[i1 + j * self.RoomNum: i2 + j * self.RoomNum+1][::-1]
        for j in range(len(self.RB)):
            r = random.sample([i for i in range(self.RoomNum)], 2)
            i1 = min(r)
            i2 = max(r)
            Rever[i1 - (len(self.RB) - j) * self.RoomNum:i2 - (len(self.RB) - j) * self.RoomNum + 1 ] =\
                T[i1 - (len(self.RB) - j) * self.RoomNum:i2 - (len(self.RB) - j) * self.RoomNum + 1 ][::-1]
        return Rever

    def mainrun(self):
        Solu = self.initSolution()
        Result = self.decode(Solu)
        Cost = self.parameter(Result)
        T = self.T0

        cnt = 1
        minCost = Cost
        minSolution = Solu
        costArray = np.zeros(self.maxIterate)
        startime = time.time()
        while T>self.Ts:
            for k in range(self.Lk):
                mode = random.randint(1,3)
                newSolu = self.createNeibor(Solu, mode)
                newResult = self.decode(newSolu)
                newCost = self.parameter(newResult)
                # import pdb;pdb.set_trace()
                delta = newCost - Cost
                if (delta<0):
                    Cost = newCost
                    Solu = newSolu
                else:
                    p = math.exp(-delta/T)
                    if random.random() <=p:
                        Cost = newCost
                        Solu = newSolu
            costArray[cnt] = Cost
            # print("迭代次数为: {}. 最优成本为: {}. 当前成本为:{}. T:{}".format(str(cnt), str(minCost), str(Cost),str(T)))
            # print(Solu)
            if Cost < minCost:
                minCost = Cost
                minSolution = Solu
            T = T*self.r
            cnt = cnt + 1
        endtime = time.time()
        return minSolution, minCost, startime-endtime

    def robustrun(self, robustD):
        self.Service = robustD
        Solu = self.initSolution()
        Result = self.decode(Solu)
        Cost = self.parameter(Result)
        T = self.T0

        cnt = 1
        minCost = Cost
        minSolution = Solu
        costArray = np.zeros(self.maxIterate)
        startime = time.time()
        while T>self.Ts:
            for k in range(self.Lk):
                mode = random.randint(1,3)
                newSolu = self.createNeibor(Solu, mode)
                newResult = self.decode(newSolu)
                newCost = self.parameter(newResult)
                # import pdb;pdb.set_trace()
                delta = newCost - Cost
                if (delta<0):
                    Cost = newCost
                    Solu = newSolu
                else:
                    p = math.exp(-delta/T)
                    if random.random() <=p:
                        Cost = newCost
                        Solu = newSolu
            costArray[cnt] = Cost
            # print("迭代次数为: {}. 最优成本为: {}. 当前成本为:{}. T:{}".format(str(cnt), str(minCost), str(Cost),str(T)))
            # print(Solu)
            if Cost < minCost:
                minCost = Cost
                minSolution = Solu
            T = T*self.r
            cnt = cnt + 1
        endtime = time.time()
        return minSolution, minCost, startime-endtime





if __name__ == '__main__':
    fold_path = './data/set5/'
    depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
    result = {}
    for i in list(depotDic.keys()):
        print(i)
        NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B = processSet2(fold_path,str(i), cal=False)
        Solver = SA(int(len(NJobs)/2), {"R1":0, "R2":0, "R3":1, "R4":1}, RB=[2,2], jobTypeNum=2, Service=D, Traveltime=traveltime, Agents=Agents)
        result[i] = Solver.mainrun()


    fold_path = './data/set1/'
    depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
    result = {}
    for i in list(depotDic.keys()):
        print(i)
        NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B = processSet1(fold_path,str(i), cal=False)
        Solver = SA(int(len(NJobs)/2), {"R1":0, "R3":1}, RB=[1,1], jobTypeNum=2, Service=D, Traveltime=traveltime, Agents=Agents)
        result[i] = Solver.mainrun()

