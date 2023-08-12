import os
import math
import time
import cmath
import random
import numpy as np
import gurobipy as gp
from collections import defaultdict
import matplotlib.pyplot as plt
from dataloader import get_data, processSet1, processSet2
plt.rcParams['font.sans-serif']=['Times New Roman']

class GurobiSolver:
    def __init__(self, NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B ):
        self.NJobs       = NJobs
        self.Agents      = Agents
        self.D           = D
        self.traveltime  = traveltime
        self.numroom     = numroom
        self.Jobs1       = Jobs1
        self.Jobs0       = Jobs0
        self.Jobs        = Jobs
        self.P           = P
        self.B           = B         

    def Uncertainty(self, Dmatrix, scenaNum=5, ratio=0.05):
        BoxD = defaultdict(lambda: 0)
        ConvexD = defaultdict(lambda: 0)
        EllipsoidalD = defaultdict(lambda: 0)
        for (i,j), value in Dmatrix.items():
            border = int(value*ratio)
            BoxLength=[]
            ConvexLength = []
            EllipsoidalLength=0
            for s in range(scenaNum):
                rand = random.randint(-border, border)
                BoxLength.append(math.fabs(rand).real)
                EllipsoidalLength += rand ** 2
                ConvexLength.append(rand)
            BoxD[i,j] = sum(BoxLength)
            ConvexD[i,j] = max(ConvexLength)
            EllipsoidalD[i,j] = math.sqrt(EllipsoidalLength).real
        return BoxD, ConvexD, EllipsoidalD

    def run(self):
        BoxD, ConvexD, EllipsoidalD = self.Uncertainty(Dmatrix=self.D, scenaNum=5, ratio=0.15)
        # defaultdict(lambda: 0)    BoxD, ConvexD, EllipsoidalD
        uncertainD = defaultdict(lambda: 0)
        model = gp.Model("vrp")
        X = model.addVars(self.Jobs, self.Jobs, self.Agents, lb=0, ub=1, vtype=gp.GRB.BINARY, name="X") # xijk
        Y = model.addVars(self.Jobs, self.Agents, lb=0, ub=1, vtype=gp.GRB.BINARY, name="Y") # ysk
        U = model.addVars(self.Jobs, vtype=gp.GRB.CONTINUOUS, name="U")
        Cmax = model.addVar(lb=0, ub=100000, vtype=gp.GRB.INTEGER, name="C_max")

        Center = "depot"
        for job in self.NJobs:
            for agent in self.Agents:
                if self.B[job, agent] == 0:
                    model.addConstr(Y[job, agent] == 0)
        model.addConstr(gp.quicksum([Y[Center, k] for k in self.Agents]) == len(self.Agents))
        for i in self.NJobs:
            model.addConstr(gp.quicksum(Y[i, k] for k in self.Agents) == 1)

        for k in self.Agents:
            model.addConstr(gp.quicksum([X[i, Center, k] for i in self.NJobs]) == 1)
        for k in self.Agents:
            for j in self.NJobs:
                model.addConstr(gp.quicksum(X[i,j,k] for i in self.Jobs) == Y[j,k])
        for k in self.Agents:
            for i in self.NJobs:
                model.addConstr(gp.quicksum(X[i, j, k] for j in self.Jobs) == Y[i, k])

        # 5 确定两个任务执行时间
        for k in self.Agents:
            for i in self.Jobs:
                for j in self.Jobs:
                    if self.B[j, k]:
                        model.addConstr(U[i] + self.D[i, k] + uncertainD[i,k] + self.traveltime[i, j] <= U[j] + 100000*(1-X[i,j,k]))

        for (j, k), prec in self.P.items():
            if prec > 0:  # if j precedes k in the DAG
                for a in self.Agents:
                    for b in self.Agents:
                        if self.B[j, a] and self.B[k, b]:
                            model.addConstr(U[k] >= U[j])
                            model.addConstr(U[k] >= (U[j] + (self.D[j, a] + uncertainD[j,a] + self.traveltime[j, k]) * (Y[j, a] + Y[k, b] - 1)))

        for k in self.Agents:
            for i in self.Jobs1:
                model.addConstr(Cmax >= U[i] + self.D[i, k] + uncertainD[i,k] + self.traveltime[i, Center]*X[i,Center,k])

        model.setObjective(Cmax, gp.GRB.MINIMIZE)
        #import pdb;pdb.set_trace()
        model.setParam("MIPFocus", 3)
        startime = time.time()
        model.optimize()
        endtiem = time.time()
        non_zero_edges = [e for e in X if X[e].X != 0]
        tours = []
        for k in self.Agents:
            tempedge = [e for e in non_zero_edges if e[-1] == k]
            parent = Center
            edge = []
            while len(tempedge) != 0:
                for e in tempedge:
                    if e[0] == parent:
                        edge.append(e)
                        tempedge.remove(e)
                        parent = e[1]
            tours.append(edge)

        return Cmax.X, endtiem-startime, tours


if __name__ == '__main__':
    fold_path = './data/set3/'
    depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
    # np.save(os.path.join(fold_path, "depot.npy"), depotDic)
    result = {}
    for i in list(depotDic.keys()):
        # if os.path.exists('{}_distance.npy'.format(i)):
        #     continue
        print(i)
        NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B = processSet2(fold_path,str(i), cal=False)
        Solver = GurobiSolver(NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B)
        result[i] = Solver.run()


    fold_path = './data/set3/'
    depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
    NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B = processSet2(fold_path, str(31887483), cal=False)
    Solver = GurobiSolver(NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B)
    a,b,c= Solver.run()

    fold_path = './data/set1/'
    depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
    NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B = processSet1(fold_path, str(45775501), cal=False)
    Solver = GurobiSolver(NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B)
    a,b,c= Solver.run()


    fold_path = './data/set1/'
    depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
    result = {}
    for i in list(depotDic.keys()):
        print(i)
        NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B = processSet1(fold_path, str(i), cal=False)
        Solver = GurobiSolver(NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B)
        result[i] = Solver.run()






# if os.path.exists('{}_distance.npy'.format(i)):
#     continue
#
#
# non_zero_edges = [ e for e in X if X[e].X != 0 ]
# tours=[]
# for k in Agents:
#     tempedge = [e for e in non_zero_edges if e[-1] == k]
#     parent = Center
#     edge = []
#     while len(tempedge)!=0:
#         for e in tempedge:
#             if e[0] == parent:
#                 edge.append(e)
#                 tempedge.remove(e)
#                 parent = e[1]
#     tours.append(edge)

# timeline = {}
# for tour in tours:
#     sequence = ['depot']
#     starttime = [0.0]
#     duringtime = [0]
#     travel = []
#     for i, j, k in tour:
#         sequence.append(j)
#         starttime.append(U[j].X)
#         duringtime.append(D[j,k])
#         travel.append(traveltime[i,j])
#     # travel.append(traveltime[j, "depot"])
#     timeline[k] = {"sequence": sequence,
#                    "starttime": starttime,
#                    "duringtime": duringtime,
#                    "traveltime": travel,}
#
# robot = "R1"
# length = len(timeline[robot]["sequence"])
# for i in range(length-1):
#     plt.barh(i, timeline[robot]["duringtime"][i], left=timeline[robot]["starttime"][i])
#     if timeline[robot]["duringtime"][i]:
#         plt.text(timeline[robot]["starttime"][i]+1, i, timeline[robot]["sequence"][i][:5], color="black", size=20)
#     if i < length-1:
#         plt.barh(i, timeline[robot]["traveltime"][i], left=timeline[robot]["starttime"][i]+timeline[robot]["duringtime"][i],color=['gray'])
# plt.show()
