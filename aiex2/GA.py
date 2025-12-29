from otsu import OTSU
import numpy as np


class GA:  # 定义遗传算法类
    def __init__(self, image, M):  # 构造函数,进行初始化以及编码
        self.image = image
        self.M = M  # 初始化种群的个体数
        self.length = 8  # 每条染色体基因长度为8(0-255)
        self.species = np.random.randint(0, 256, self.M)  # 给种群随机编码
        self.select_rate = 0.5  # 选择的概率
        self.strong_rate = 0.3  # 选择适应性强染色体的比率
        self.bianyi_rate = 0.05  # 变异的概率

    def Adaptation(self, ranseti):  # 进行染色体适应度的评估
        fit = OTSU().otsu(self.image, ranseti)
        return fit

    def selection(self):  # 进行个体的选择
        fitness = []
        for ranseti in self.species:  # 循环遍历种群，计算适应度
            fitness.append((self.Adaptation(ranseti), ranseti))
        
        # 逆序排序，适应度高的染色体排前面
        fitness1 = sorted(fitness, key=lambda x: x[0], reverse=True)
        
        # 提取排序后的染色体
        sorted_species = [x[1] for x in fitness1]
        
        # 适应性特别强的直接留下来
        num_strong = int(len(sorted_species) * self.strong_rate)
        parents = sorted_species[:num_strong]
        
        # 挑选适应性没那么强的染色体
        for ranseti in sorted_species[num_strong:]:
            if np.random.random() < self.select_rate:
                parents.append(ranseti)
        return parents

    def crossover(self, parents):  # 进行个体的交叉
        children = []
        child_count = len(self.species) - len(parents)  # 补足种群数量（或者说产生新个体）
        
        while len(children) < child_count:
            father_idx = np.random.randint(0, len(parents))
            mather_idx = np.random.randint(0, len(parents))
            
            if father_idx != mather_idx:
                father = parents[father_idx]
                mather = parents[mather_idx]
                
                position = np.random.randint(0, self.length)  # 随机选取交叉位置
                mask = 0
                for i in range(position):  # 11110000
                    mask = mask | (1 << i)
                
                child = (father & mask) | (mather & ~mask)
                children.append(child)
        
        self.species = parents + children  # 更新种群

    def bianyi(self):  # 进行个体的变异（变异策略可以自己调整，细微影响收敛的快慢）
        for i in range(len(self.species)):
            if np.random.random() < self.bianyi_rate:
                j = np.random.randint(0, self.length)
                self.species[i] = self.species[i] ^ (1 << j)  # 亦或加移位

    def evolution(self):  # 进行个体的进化
        parents = self.selection()
        self.crossover(parents)
        self.bianyi()

    def get_threshold(self):  # 返回适应度最高的染色体
        fitness = []
        for ranseti in self.species:
            fitness.append((self.Adaptation(ranseti), ranseti))
        # 返回适应度最高的那个染色体的值
        return max(fitness, key=lambda x: x[0])[1]