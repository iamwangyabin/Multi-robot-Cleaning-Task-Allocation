import os
import math
import random
import numpy as np
from collections import defaultdict
from dataloader import get_data, processSet1, processSet2
from Gurobi import GurobiSolver
from GA import GA as GASolver
from SA import SA as SASolver
from PSO import Model as PSOSolver

def Uncertainty(Dmatrix, scenaNum=5, ratio=0.05):
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
        BoxD[i,j] = sum(BoxLength)+Dmatrix[i,j]
        ConvexD[i,j] = max(ConvexLength)+Dmatrix[i,j]
        EllipsoidalD[i,j] = math.sqrt(EllipsoidalLength).real+Dmatrix[i,j]
    return BoxD, ConvexD, EllipsoidalD

result = {}

fold_path = './data/set5/'
depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
for i in list(depotDic.keys()):
    NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B = processSet2(fold_path, str(i), cal=False)
    for ritho in [0.05,0.10,0.15]:
        print(str(i))
        Solver = SASolver(int(len(NJobs)/2), {"R1":0, "R2":0, "R3":1, "R4":1}, RB=[2,2], jobTypeNum=2, Service=D, Traveltime=traveltime, Agents=Agents)
        BoxD, ConvexD, EllipsoidalD = Uncertainty(D, scenaNum=5, ratio=ritho)
        result[(i, "box", ritho)] = Solver.robustrun(BoxD)
        result[(i, "conv", ritho)] = Solver.robustrun(ConvexD)
        result[(i, "elli", ritho)] = Solver.robustrun(EllipsoidalD)


fold_path = './data/set1/'
depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
for i in list(depotDic.keys()):
    NJobs, Agents, D, traveltime, numroom, Jobs1, Jobs0, Jobs, P, B = processSet1(fold_path, str(i), cal=False)
    for ritho in [0.05,0.10,0.15]:
        print(str(i))
        Solver = SASolver(int(len(NJobs)/2), {"R1":0, "R3":1}, RB=[1,1], jobTypeNum=2, Service=D, Traveltime=traveltime, Agents=Agents)
        BoxD, ConvexD, EllipsoidalD = Uncertainty(D, scenaNum=5, ratio=ritho)
        result[(i, "box", ritho)] = Solver.robustrun(BoxD)
        result[(i, "conv", ritho)] = Solver.robustrun(ConvexD)
        result[(i, "elli", ritho)] = Solver.robustrun(EllipsoidalD)





fold_path = './data/set5/'
depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()
roomnum = []
areas = []
for i in list(depotDic.keys()):
    scaleDic = np.load("scale.npy", allow_pickle=True).item()
    scale = scaleDic[str(i)]
    if os.path.exists('{}_distance.npy'.format(i)):
        map2d, distance, centroid = get_data(fold_path=fold_path, imgpath=str(i), cal=False)
    else:
        map2d, distance, centroid = get_data(fold_path = fold_path, imgpath=str(i), cal=True)
    roomnum.append(len(centroid))
    area = 0
    for key,value in centroid.items():
        area+=value['num']
    areas.append(area*(scale**2))

print(max(areas))
print(min(areas))
print(sum(areas)/len(areas))
print(max(roomnum))
print(min(roomnum))
print(sum(roomnum)/len(roomnum))

import openpyxl

def write_excel_xlsx(path, sheet_name, value):
    index = len(value)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.cell(row=i+1, column=j+1, value=str(value[i][j]))
    workbook.save(path)

robust  = np.load( "robust_set1.npy", allow_pickle=True).item()

# excel
fold_path = './data/set1/'
depotDic = np.load(os.path.join(fold_path, "depot.npy"), allow_pickle=True).item()

book_name_xlsx = 'set1.xlsx'
sheet_name_xlsx = 'xlsx格式测试表'

# id box box box conv conv conv elli elli elli
value = []
for i in list(depotDic.keys()):
    temp = []
    temp.append(str(i))
    temp.append(str(robust[(i,'box', 0.05)][1]))
    temp.append(str(robust[(i,'box', 0.10)][1]))
    temp.append(str(robust[(i,'box', 0.15)][1]))
    temp.append(str(robust[(i,'conv', 0.05)][1]))
    temp.append(str(robust[(i,'conv', 0.10)][1]))
    temp.append(str(robust[(i,'conv', 0.15)][1]))
    temp.append(str(robust[(i,'elli', 0.05)][1]))
    temp.append(str(robust[(i,'elli', 0.10)][1]))
    temp.append(str(robust[(i,'elli', 0.15)][1]))
    value.append(temp)

write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, value)



def sqrt(x,):
    if x<0:
        return 0
    else:
        low=0
        up=1000
        mid = (low+up)/2
        lae = low
        while mid-lae>0.00001:
            if mid*mid>x:
                up=mid
            else:
                low=mid
            mid=(up+low)/2
            lae=low
    return mid


seq = [9,9,6,0,9]
def sqrt(seq, k):
    maxleng = 0
    leng = len(seq)
    for i in range(0, leng):
        templeng = 0
        for j in range(i+1, leng):
            if len([ii for ii in seq[i:j+1] if ii > k]) > len([ii for ii in seq[i:j+1] if ii <= k]):
                templeng = j-i+1
            if templeng > maxleng:
                maxleng=templeng
    return maxleng

print(sqrt(seq, 8))

import sys
line = sys.stdin.readline().strip().split()
second = int(line[1])
line = sys.stdin.readline().strip().split()
seq=[int(x) for x in line]
print(sqrt(seq, second))


def sieve(subweigh):
    avgs = int(sum(subweigh)/len(subweigh))
    return [ii for ii in subweigh if ii > avgs], [ii for ii in subweigh if ii <= avgs]
stact = []
def orange(weight):
    stact.append(sum(weight))
    if len(weight) != 0:
        currentweight = weight.copy()
        w1, w2 = sieve(currentweight)
        if len(w1) > 1 :
            orange(w1)
        if len(w2) > 1:
            orange(w2)
        return sum(currentweight)
    else:
        return 0


import sys
line = sys.stdin.readline().strip().split()
second = int(line[1])
line = sys.stdin.readline().strip().split()
weight=[int(x) for x in line]
s = []
for i in range(second):
    s.append(int(sys.stdin.readline().strip()))
for i in s:
    if i in stact:
        print("YES")
    else:
        print("NO")

print(sqrt(seq, second))
weight=[7,2,1,6,5]
_ = orange(weight)
#输入  s
s=[3,21,30]
