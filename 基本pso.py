# coding: utf-8
import numpy as np
import random
import matplotlib.pyplot as plt
class PSO():
    def __init__(self, pN, dim, max_iter):
        self.w = 0.5
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.4
        self.r2 = 0.3
        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度
        self.cv = np.random.randint(500,1000) #cv表示任务v所需要的总的cpu数,单位MHZ
        self.fl = np.random.uniform(500,1000) #fl表示移动终端本地计算能力,单位MHZ
        self.fc = 10000  #fc表示MEC服务器的计算能力，单位为MHz
        self.k = 10**-26 #微型集成电路架构的一个系数
        self.d = np.random.randint(100,3000) #从本地传入到MEC服务器的数据的大小,单位为KB/S
        self.Ru = 256 #移动终端上行电路发行功率,单位为KB/S
        self.pi = 0.5 #移动终端的发送功率,单位为w
        self.piI = 0.01 #移动设备空闲时的功耗,单位为w
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置
        self.V = np.zeros((self.pN, self.dim)) #所有粒子的速度
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置
        self.gbest = np.zeros((1, self.dim))    #全局最佳位置
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 1e10  # 全局最佳适应值
        self.Sj=0 #决策因子，或0或1

    def function(self, x):
        sum = 0
        length = len(x)
        x = x ** 2
        for i in range(length):
            sum += x[i]
        return sum
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
              # 输出最优值
        return fitness
my_pso=PSO(pN=30, dim=10, max_iter=100)
my_pso.init_Population()
fitness = my_pso.iterator()
 #-------------------画图--------------------
plt.figure(1)
plt.title("handsomehua")
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size=14)
t = np.array([t for t in range(0, 100)])
fitness = np.array(fitness)
plt.plot(t, fitness, color='red', linewidth=3)
plt.show()