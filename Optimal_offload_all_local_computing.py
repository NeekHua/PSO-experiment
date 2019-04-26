# -*-coding:utf-8 -*-
# coding: utf-8
from pandas import DataFrame,Series
import numpy as np
import random
import matplotlib.pyplot as plt
# ----------------------PSO参数设置---------------------------------
class PSO():
    def __init__(self,pN,dim,max_iter):
        self.pN=pN
        self.w = 0.5
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.4
        self.r2 = 0.3
        self.dim = dim  # 搜索维度
        self.ci = np.random.uniform(0.5,1,30) #cv表示任务v所需要的总的cpu数,共有200个任务,单位MHZ
        self.fl = np.random.uniform(0.5,1) #fl表示移动终端本地计算能力,单位MHZ
        self.fi = 10  #fc表示MEC服务器的计算能力，单位为MHz
        self.k = 10**-26 #微型集成电路架构的一个系数
        self.di = np.random.randint(100,3000,30) #从本地传入到MEC服务器的数据的大小,设有30个任务的数据大小,单位为KB/S
        self.ru = 256 #移动终端上行电路发送速率,单位为KB/S
        self.pi = 0.5 #移动终端的发送功率,单位为w
        self.piI = 0.01 #移动设备空闲时的功耗,单位为w
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置
        self.V = np.zeros((self.pN, self.dim)) #所有粒子的速度
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置
        self.gbest = np.zeros((1, self.dim))#全局最佳位置
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 1e10  # 全局最佳适应值
        #self.Sj=np.random.randint(0,2,30) #决策因子,随机取30个0,1的数

    # ---------------------目标函数Sphere函数-----------------------------
    def function(self, x):
        sum = 0
        length = len(x)
        #x = x ** 2
        for i in range(length):
            sum += x[i]
        return sum
    # ---------------------初始化种群----------------------------------
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = random.uniform(0, 1)
                self.V[i][j] = random.uniform(0, 1)
            self.pbest[i] = self.X[i]
            tmp = self.function(self.X[i])
            self.p_fit[i] = tmp
            if (tmp < self.fit):
                self.fit = tmp
                self.gbest = self.X[i]

            # ----------------------更新粒子位置----------------------------------
    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            for i in range(self.pN):  # 更新gbest\pbest
                temp = self.function(self.X[i])
                if (temp < self.p_fit[i]):  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if (self.p_fit[i] < self.fit):  # 更新全局最优
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
            fitness.append(self.fit)
        #print(self.fit)  # 输出最优值
        return fitness,self.V,self.X
#计算任务在本地执行所需的时间以及任务在本地执行所产生的能耗
    def local_time_energy(self):
        ci=self.ci
        # 从500,1000中随机选取一个浮点数
        fl=self.fl
        local_time = ci / fl
        k=10**-26
        local_energy =k*ci*(fl**2)
        return local_time,local_energy
        # -------------------计算任务卸载在MEC服务器计算所需的时间
    # 以及任务在MEC服务器上执行所产生的能耗----------------------
    def cloud_server_energy_time(self):
        di=self.di
        ci=self.ci
        ru=self.ru
        fi=self.fi
        pi=self.pi
        piI=self.piI
        cloud_server_time=di/ru+ci/fi
        cloud_server_energy=pi*(di/ru)+piI*(ci/fi)
        return cloud_server_time,cloud_server_energy
    def total_energy(self):
    #pN个任务在本地执行所产生的能耗和在MEC服务器上能耗之和
        Sj=self.sigmoid_function()
        sum=0
        lte = self.local_time_energy()[1]
        cset = self.cloud_server_energy_time()[1]
        for i,j,z in zip(Sj,lte,cset):
            a=((1-i)*j)+(i*z)
            sum+=a
        return sum
        #for a in i:
        #    sum += a
        #return sum
    #贪婪算法求得的总能量
    def greedy_algorithm_total_energy(self):
        Sj=self.greedy_algorithm()
        sum=0
        lte = self.local_time_energy()[1]
        cset = self.cloud_server_energy_time()[1]
        for i, j, z in zip(Sj, lte, cset):
            a = ((1 - i) * j) + (i * z)
            sum += a
        return sum
#运用贪婪算法取得决策因子，并将取得的决策因子带入总能量函数中
    def greedy_algorithm(self):
        list_greedy_algorithm=[]
        Sj=np.random.uniform(0,1,self.pN*self.dim)
        p=np.random.uniform(0,0.5,self.pN*self.dim)
        for (p,Sj) in zip(p,Sj):
            if p<Sj:
                greedy=1
            else:
                greedy=0
            list_greedy_algorithm.append(greedy)
        Sj=list_greedy_algorithm
        return Sj
    #找出决策因子应该取0还是1
    def sigmoid_function(self):
        list=[]
        V=self.iterator()[1]
        Svij=1/(1+np.exp(-V))
        p=np.random.uniform(0,1,self.pN*self.dim)
        for (p,Svij) in zip(p,Svij):
            if p < Svij:
                S = 1
            else:
                S=0
            list.append(S)
        Sj=list
        return Sj
    #----------------------程序执行-----------------------
list=[]
for j in range(5,35,5):
    my_pso = PSO(pN=j, dim=1, max_iter=100)
    my_pso.init_Population()
    c=my_pso.total_energy()
    list.append(c)
print(list)

#-------------------画图--------------------
plt.figure(1)
#plt.title("particle-total_energy")
plt.xlabel("Number of MTs", size=14)
plt.ylabel("Energy consumption(J)", size=14)
t =[5,10,15,20,25,30]
plt.grid()
plt.plot(t,list,color='blue',linewidth=2)
plt.plot(t,local,color='green',linewidth=2)
plt.plot(t,mec,color='red',linewidth=2 )
#plt.plot(t, compare, color='blue', linewidth=3)
#plt.plot(t,optimal, color='red', linewidth=3)
plt.legend(['Optimal offloading','All_local_computing','All_MEC_computing'],loc = 'upper left')
plt.show()


