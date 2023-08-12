import os
import copy
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from dataloader import get_data, processSet1, processSet2

class Sol():
    def __init__(self):
        self.nodes_seq=None
        self.obj=None

class Node():
    def __init__(self):
        self.id=0
        self.name=''
        self.seq_no=0
        self.x_coord=0
        self.y_coord=0
        self.demand=0

class Model():
    def __init__(self, RoomNum, robot_fix, RB, jobTypeNum, Service, Traveltime, Agents, epochs=1000,popsize=2000,Vmax=2,w=0.9,c1=1,c2=5):
        """
         :param epochs: Iterations
         :param popsize: Population size
         :param Vmax: Max speed
         :param w: Inertia weight
         :param c1:Learning factors
         :param c2:Learning factors
         """
        self.Vmax = Vmax
        self.w=w
        self.c1=c1
        self.c2=c2
        self.epochs = epochs
        self.popsize = popsize
        self.RoomNum = RoomNum
        self.robot_fix = robot_fix
        self.RB = RB
        self.jobTypeNum = jobTypeNum
        self.Service = Service
        self.Traveltime = Traveltime
        self.Agents = Agents
        self.number_of_nodes=len(self.randomGenerateT())
        self.sol_list=[]
        self.best_sol=None
        self.pl=[]
        self.pg=None
        self.v=[]

    def randomGenerateT(self):
        T=[]
        for i in range(self.jobTypeNum):
            tempjob=[]
            for i in range(self.RoomNum):
                tempjob.append((i+1))
            random.shuffle(tempjob)
            T.extend(tempjob)
        for j in self.RB:
            remain = self.RoomNum
            for i in range(j-1):
                temp = random.randint(1,remain-1)
                remain = remain - temp
                T.append(temp)
            T.append(remain)
        return T

    def decode(self, T):
        T = T.copy()
        robotnum=len(self.Agents)
        jobsplit=T[-robotnum:]
        Joblist = {}
        p=0
        for i in range(len(self.Agents)):
            Joblist[self.Agents[i]] = T[p:p+jobsplit[i]]
            p = p + jobsplit[i]
        # 任务是有顺序的
        # 首先把每个路径都提出来单独计算，如果有冲突二次调整，对于T 我们假定任务序列前面的比后面的要先完成，因此使用前面的值计算后面的即可
        Result = []
        prejobTime = defaultdict(lambda: 0)
        for (robot, jobs) in Joblist.items():
            if self.robot_fix[robot] == 0:
                temptime = 0
                pre="depot"
                for j in jobs:
                    temptime = temptime + self.Traveltime[pre, "room"+str(j)+"_"+str(self.robot_fix[robot])] + self.Service["room"+str(j)+"_"+str(self.robot_fix[robot]), robot]
                    Result.append([j, robot, temptime])
                    prejobTime["room"+str(j)] = temptime
                    pre = "room"+str(j)+"_"+str(self.robot_fix[robot])
            if self.robot_fix[robot] == 1:
                temptime = 0
                pre="depot"
                for j in jobs:
                    if temptime + self.Traveltime[pre, "room"+str(j)+"_"+str(self.robot_fix[robot])] > prejobTime["room"+str(j)]:
                        temptime = temptime + self.Traveltime[pre, "room"+str(j)+"_"+str(self.robot_fix[robot])] + self.Service["room"+str(j)+"_"+str(self.robot_fix[robot]), robot]
                    else:
                        temptime = temptime + prejobTime["room"+str(j)] + self.Service["room"+str(j)+"_"+str(self.robot_fix[robot]), robot]
                    Result.append([j, robot, temptime])
                    prejobTime["room"+str(j)] = temptime
                    pre = "room"+str(j)+"_"+str(self.robot_fix[robot])
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
                if Cmax < pretime+self.Traveltime["room"+str(prejob)+"_0", "depot"]:
                    Cmax = pretime+self.Traveltime["room"+str(prejob)+"_0", "depot"]
                pre = i[1]
                pretime = i[2]
                prejob = i[0]
        if Cmax < pretime + self.Traveltime["room" + str(prejob) + "_0", "depot"]:
            Cmax = pretime + self.Traveltime["room" + str(prejob) + "_0", "depot"]
        return Cmax

    def genInitialSol(self, popsize):
        best_sol=Sol()
        best_sol.obj=float('inf')
        for i in range(popsize):
            tempSolu = self.randomGenerateT() # list
            sol=Sol()
            sol.nodes_seq= copy.deepcopy(tempSolu)
            newResult = self.decode(tempSolu)
            newCost = self.parameter(newResult)
            sol.obj = newCost
            self.sol_list.append(sol)
            self.v.append([self.Vmax]*self.number_of_nodes) ########这里应该是可行解长度
            self.pl.append(sol.nodes_seq)  # 记录可行解
            if sol.obj<best_sol.obj:
                best_sol=copy.deepcopy(sol)
        self.best_sol=best_sol
        self.pg=best_sol.nodes_seq

    def updatePosition(self):
        for id,sol in enumerate(self.sol_list):
            x=sol.nodes_seq
            v=self.v[id]
            pl=self.pl[id]
            r1=random.random()
            r2=random.random()
            new_v=[]
            for i in range(self.number_of_nodes):
                v_=self.w*v[i]+self.c1*r1*(pl[i]-x[i])+self.c2*r2*(self.pg[i]-x[i])
                if v_>0:
                    new_v.append(min(v_, self.Vmax))
                else:
                    new_v.append(max(v_, -self.Vmax))
            new_x=[x[i]+new_v[i] for i in range(self.number_of_nodes)]
            new_x=self.adjustCode(new_x)
            self.v[id]=new_v
            new_r = self.decode(new_x)
            new_x_obj = self.parameter(new_r)
            if new_x_obj<sol.obj:
                self.pl[id]=copy.deepcopy(new_x)
            if new_x_obj<self.best_sol.obj:
                self.best_sol.obj=copy.deepcopy(new_x_obj)
                self.best_sol.nodes_seq=copy.deepcopy(new_x)
                self.pg=copy.deepcopy(new_x)
            self.sol_list[id].nodes_seq = copy.deepcopy(new_x)
            self.sol_list[id].obj = copy.deepcopy(new_x_obj)


    def adjustCode(self, code: list):
        tempjob = []
        for i in range(self.RoomNum):
            tempjob.append((i + 1))
        for i in range(self.jobTypeNum):
            tempcode = code[i * self.RoomNum: (i + 1) * self.RoomNum].copy()
            sortedargs = sorted(range(len(tempcode)), key=lambda k: tempcode[k])  # 返回排序下标
            ii = 0
            for jj in sortedargs:
                code[ii + i * self.RoomNum] = tempjob[jj]
                ii += 1
        splitIndex = self.jobTypeNum * self.RoomNum
        for i in range(self.jobTypeNum):
            Sum = 0
            temp = []
            for j in range(self.RB[i]):
                Sum += math.fabs(code[splitIndex + i * self.RB[i] + j])
            for j in range(self.RB[i] - 1):
                temp.append(round((self.RoomNum / Sum) * math.fabs(code[splitIndex + i * self.RB[i] + j])))
            temp.append(self.RoomNum - sum(temp))
            for j in range(self.RB[i]):
                code[splitIndex + i * self.RB[i] + j] = temp[j]
        return code

    def setPara(self, epochs=1000,popsize=2000,Vmax=2,w=0.9,c1=1,c2=5):
        self.Vmax = Vmax
        self.w=w
        self.c1=c1
        self.c2=c2
        self.epochs = epochs
        self.popsize = popsize

    def run(self):
        startTime = time.time()
        history_best_obj=[]
        self.genInitialSol(self.popsize)
        history_best_obj.append(self.best_sol.obj)
        for ep in range(self.epochs):
            self.updatePosition()
            history_best_obj.append(self.best_sol.obj)
            print("%s/%s: best obj: %s"%(ep,self.epochs,self.best_sol.obj))
            # print(self.best_sol.nodes_seq)
        endTime = time.time()
        return self.best_sol, self.best_sol.obj, endTime-startTime


if __name__=='__main__':
    # fold_path = './data/set2/'
    # depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
    # NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B = processSet2(fold_path, '31450071', cal=False)
    #
    # Solver = Model(int(len(NJobs)/2), {"R1":0, "R2":0, "R3":1, "R4":1}, RB=[2,2], jobTypeNum=2, Service=D, Traveltime=traveltime, Agents=Agents)
    # Solver.setPara(epochs=1000,popsize=2000,Vmax=5,w=0.8,c1=2,c2=2)
    # Solver.run()

    fold_path = './data/set5/'
    depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
    result = {}
    for i in list(depotDic.keys()):
        print(i)
        NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B = processSet2(fold_path,str(i), cal=False)
        # Solver = GA(int(len(NJobs)/2), {"R1":0, "R3":1}, RB=[1,1], jobTypeNum=2, Service=D, Traveltime=traveltime, Agents=Agents)
        Solver = Model(int(len(NJobs) / 2), {"R1": 0, "R2": 0, "R3": 1, "R4": 1}, RB=[2, 2], jobTypeNum=2, Service=D,
                       Traveltime=traveltime, Agents=Agents)
        Solver.setPara(epochs=500, popsize=2000, Vmax=5, w=0.8, c1=2, c2=2)
        result[i] = Solver.run()


    fold_path = './data/set1/'
    depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
    result = {}
    for i in list(depotDic.keys()):
        print(i)
        NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B = processSet1(fold_path,str(i), cal=False)
        # Solver = GA(int(len(NJobs)/2), {"R1":0, "R3":1}, RB=[1,1], jobTypeNum=2, Service=D, Traveltime=traveltime, Agents=Agents)
        Solver = Model(int(len(NJobs) / 2), {"R1": 0, "R3": 1}, RB=[1, 1], jobTypeNum=2, Service=D,
                       Traveltime=traveltime, Agents=Agents)
        Solver.setPara(epochs=500, popsize=2000, Vmax=5, w=0.8, c1=2, c2=2)
        result[i] = Solver.run()