import os
import numpy as np
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from dataloader import get_data, processSet1, processSet2

class GA:
    def __init__(self, RoomNum, robot_fix, RB, jobTypeNum, Service, Traveltime, Agents):
        self.RoomNum = RoomNum
        self.robot_fix = robot_fix
        self.RB = RB
        self.jobTypeNum = jobTypeNum
        self.Service = Service
        self.Traveltime = Traveltime
        self.Agents = Agents
        self.population_num = 200 # 种群规模
        self.Pc = 0.9 # prob crosscover 交叉概率
        self.Pm = 0.08 # 变异概率
        self.Max_gen = 3000 # 迭代次数

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
        T = T.astype(int).tolist()
        robotnum=len(self.Agents) # T 后四个为机器人向量每个机器人完成的任务数量 T=[4, 3, 1, 5, 9, 7, 8, 6, 2, 8, 4, 2, 3, 9, 1, 7, 5, 6, 2, 7, 6, 3]
        jobsplit=T[-robotnum:]
        Joblist = {}  # 输出类似于 {'R1': [5, 4], 'R2': [3, 2], 'R3': [1, 6]}
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

    def cross(self, A, B):
        A_1 = A.astype(int).tolist()
        B_1 = B.astype(int).tolist()
        for ii in range(len(self.RB)):
            r = np.random.permutation([j + 1 for j in range(self.RoomNum)])
            # r = np.array([8, 9, 3, 5, 2, 1, 4, 7, 6])
            c = min(r[:2]) + ii*self.RoomNum
            d = max(r[:2]) + ii*self.RoomNum
            for i in range(c-1,d):
                A_1[i]=0
            tempB1=[]
            for b1 in B.astype(int).tolist()[ii*self.RoomNum:(ii+1)*self.RoomNum]:
                if b1 not in A_1[ii*self.RoomNum:(ii+1)*self.RoomNum]:
                    tempB1.append(b1)
            A_1[c-1:d]=tempB1
            for i in range(c-1,d):
                B_1[i]=0
            tempA1=[]
            for a1 in A.astype(int).tolist()[ii*self.RoomNum:(ii+1)*self.RoomNum]:
                if a1 not in B_1[ii*self.RoomNum:(ii+1)*self.RoomNum]:
                    tempA1.append(a1)
            B_1[c-1:d]=tempA1
        B_1[-sum(self.RB):] = A[-sum(self.RB):].astype(int).tolist()
        A_1[-sum(self.RB):] = B[-sum(self.RB):].astype(int).tolist()
        return A_1,B_1

    # 杂交函数
    def Mating_pool(self, population):
        """
        :param 输入: population, population_num, Pc
        :return: new_popopulation_intercross c3，配对池：随机将种群population两两配对 pool
        """
        pl = np.random.permutation([j+1 for j in range(self.population_num)])
        num = int(self.population_num / 2)
        c3 = np.zeros([2, int(num)])
        pool = []
        new_pop_intercross = population.copy()
        # 生成“配对池c3 配对池就是把基因两辆配对
        for kj in range(num):
            c3[0,kj]=pl[2*kj]
            c3[1,kj]=pl[2*kj+1]
        # 判断“配对池c3”每一对个体的随机数是否小于交叉概率Pc
        rd = np.random.rand(num)
        for kj in range(num):
            if rd[kj]<self.Pc:
                pool.append(c3[:,kj]-1)
               # pool=[pool,c3[:,kj]]
        # 判断配对池每一对个体的随机数是否小于交叉概率Pc,若小于，保存到“产子池pool”
        pool = np.array(pool)
        pool_num=pool.shape[0]
        for kj in range(pool_num):
            c1=population[int(pool[kj, 0]),:].copy()
            c2=population[int(pool[kj, 1]),:].copy()
            new_c1,new_c2=self.cross(c1,c2)
            # import pdb;pdb.set_trace()
            new_pop_intercross[int(pool[kj, 0]),:] = np.array(new_c1)
            new_pop_intercross[int(pool[kj, 1]),:] = np.array(new_c2)
        return new_pop_intercross

    # 变异
    def Mutation(self, Cross_Pop):
        Mut_Pop = Cross_Pop.copy()
        Cross_Pop_num=Cross_Pop.shape[0]
        for j in range(Cross_Pop_num):
            A=Cross_Pop[j,:].copy()
            A_1 = A
            for jj in range(len(self.RB)):
                r=[random.uniform(0, 1) for _ in range(self.RoomNum)]
                Pe=[ii for ii in range(len(r)) if r[ii] < self.Pm]
                sum_Pe=len(Pe)
                for i in range(sum_Pe):
                    c=A[Pe[i] +jj*self.RoomNum]
                    A_1[Pe[i] +jj*self.RoomNum]=A_1[[ii for ii in range(len(r)) if r[ii] == max(r)][0] +jj*self.RoomNum]
                    A_1[[ii for ii in range(len(r)) if r[ii] == max(r)][0] +jj*self.RoomNum]=c
            Mut_Pop = np.r_[Mut_Pop, [A_1]]
        return Mut_Pop

    def run(self):
        # 种群初始化
        population=np.zeros([self.population_num, len(self.randomGenerateT())])
        for i in range(0, self.population_num):
             population[i,:] = np.array(self.randomGenerateT())

        start_time = time.time()  # 开始时间
        y=1 # %循环计数器
        while y < self.Max_gen:
            # 交叉
            new_pop_intercross = self.Mating_pool(population)
            # 变异
            new_pop_mutation = self.Mutation(new_pop_intercross)
            # 计算目标函数
            mutation_num=new_pop_mutation.shape[0]
            Total_Dis=[]
            for k in range(mutation_num):
                Result = self.decode(new_pop_mutation[k,:])
                Total_Dis.append(self.parameter(Result))
            new_pop_new = np.zeros([self.population_num, len(self.randomGenerateT())])
            Total_Dissort=np.array(Total_Dis).sort()
            index=np.array(Total_Dis).argsort()
            for k in range(self.population_num):
                new_pop_new[k,:]=new_pop_mutation[index[k],:]
            population = new_pop_new
            y = y + 1
        end_time = time.time()  # 结束时间
        Dis_min1=min(Total_Dis)
        for k in range(mutation_num):
            if Total_Dis[k]==Dis_min1:
                position1= k
                break
        X_Best=new_pop_mutation[position1,:]
        Y_Obj=Total_Dis[position1]
        return X_Best, Y_Obj, end_time-start_time

if __name__ == '__main__':
    fold_path = './data/set5/'
    depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
    result = {}
    for i in list(depotDic.keys()):
        print(i)
        NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B = processSet2(fold_path,str(i), cal=False)
        # Solver = GA(int(len(NJobs)/2), {"R1":0, "R3":1}, RB=[1,1], jobTypeNum=2, Service=D, Traveltime=traveltime, Agents=Agents)
        Solver = GA(int(len(NJobs)/2), {"R1":0, "R2":0, "R3":1, "R4":1}, RB=[2,2], jobTypeNum=2, Service=D, Traveltime=traveltime, Agents=Agents)
        result[i] = Solver.run()


    fold_path = './data/set1/'
    depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
    result = {}
    for i in list(depotDic.keys()):
        print(i)
        NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B = processSet1(fold_path,str(i), cal=False)
        # Solver = GA(int(len(NJobs)/2), {"R1":0, "R3":1}, RB=[1,1], jobTypeNum=2, Service=D, Traveltime=traveltime, Agents=Agents)
        Solver = GA(int(len(NJobs)/2), {"R1":0, "R3":1}, RB=[1,1], jobTypeNum=2, Service=D, Traveltime=traveltime, Agents=Agents)
        result[i] = Solver.run()


    fold_path = './data/set2/'
    depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
    NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B = processSet2(fold_path, '31450071', cal=False)
    Solver = GA(int(len(NJobs) / 2), {"R1": 0, "R2": 0, "R3": 1, "R4": 1}, RB=[2, 2], jobTypeNum=2, Service=D,
                Traveltime=traveltime, Agents=Agents)
    result = Solver.run()


    np.save("ga_set1.npy", result)