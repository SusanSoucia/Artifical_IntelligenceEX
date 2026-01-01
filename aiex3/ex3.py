import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import tkinter as tk
from tkinter import messagebox
import tkinter.font as tkFont

# mpl.rcParams['font.sans-serif'] = ['SimHei']  # Add this to display Chinese in graphs


class PSO_VRP:
    def __init__(self, birdNum=50, w=0.2, c1=0.4, c2=0.4, iterMax=50, CAPACITY=120, DISTABCE=250, C0=30, C1=1):
        self.birdNum = birdNum
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.iterMax = iterMax
        self.CAPACITY = CAPACITY
        self.DISTABCE = DISTABCE
        self.C0 = C0
        self.C1 = C1
        self.Customer = [(50, 50), (96, 24), (40, 5), (49, 8), (13, 7), (29, 89), (48, 12), (84, 39), (14, 47), (2, 24), (3, 82),(65, 10), (98, 52), (84, 25), (41, 69), (1, 65), (50, 71), (75, 83), (29, 32), (99, 3), (50, 93), (80, 94),(5, 42), (62, 70), (31, 62), (20, 97), (91, 75), (27, 49), (23, 15), (20, 70), (85, 60), (98, 85)]
        self.Demand = [0, 16, 11, 6, 10, 7, 12, 16, 6, 16, 8,14, 7, 16, 3, 22, 1, 19, 18, 14, 8, 12,4, 8, 24, 24, 2, 10, 15, 2, 14, 9]
        self.dis_matrix = self.calDistance(self.Customer)

    def calDistance(self, CityCoordinates):
        dis_matrix = pd.DataFrame(data=None, columns=range(len(CityCoordinates)), index=range(len(CityCoordinates)))
        for i in range(len(CityCoordinates)):
            xi, yi = CityCoordinates[i][0], CityCoordinates[i][1]
            for j in range(len(CityCoordinates)):
                xj, yj = CityCoordinates[j][0], CityCoordinates[j][1]
                dis_matrix.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 2)
        return dis_matrix

    def greedy(self, CityCoordinates, dis_matrix):
        dis_matrix = dis_matrix.astype('float64')
        for i in range(len(CityCoordinates)): dis_matrix.loc[i, i] = math.pow(10, 10)
        dis_matrix.loc[:, 0] = math.pow(10, 10)
        line = []
        now_city = random.randint(1, len(CityCoordinates) - 1)
        line.append(now_city)
        dis_matrix.loc[:, now_city] = math.pow(10, 10)
        for i in range(1, len(CityCoordinates) - 1):
            next_city = dis_matrix.loc[now_city, :].idxmin()
            line.append(next_city)
            dis_matrix.loc[:, next_city] = math.pow(10, 10)
            now_city = next_city
        return line

    def calFitness(self, birdPop, Demand, dis_matrix, CAPACITY, DISTABCE, C0, C1):
        birdPop_car, fits = [], []
        for j in range(len(birdPop)):
            bird = birdPop[j]
            lines = []
            line = [0]
            dis_sum = 0
            dis, d = 0, 0
            i = 0
            while i < len(bird):
                if line == [0]:
                    dis += dis_matrix.loc[0, bird[i]]
                    line.append(bird[i])
                    d += Demand[bird[i]]
                    i += 1
                else:
                    if (dis_matrix.loc[line[-1], bird[i]] + dis_matrix.loc[bird[i], 0] + dis <= DISTABCE) & (
                            d + Demand[bird[i]] <= CAPACITY):
                        dis += dis_matrix.loc[line[-1], bird[i]]
                        line.append(bird[i])
                        d += Demand[bird[i]]
                        i += 1
                    else:
                        dis += dis_matrix.loc[line[-1], 0]
                        line.append(0)
                        dis_sum += dis
                        lines.append(line)
                        dis, d = 0, 0
                        line = [0]
            dis += dis_matrix.loc[line[-1], 0]
            line.append(0)
            dis_sum += dis
            lines.append(line)
            birdPop_car.append(lines)
            fits.append(round(C1 * dis_sum + C0 * len(lines), 1))
        return birdPop_car, fits

    def crossover(self, bird, pLine, gLine, w, c1, c2):
        croBird = [None] * len(bird)
        parent1 = bird
        randNum = random.uniform(0, sum([w, c1, c2]))
        if randNum <= w:
            parent2 = [bird[i] for i in range(len(bird) - 1, -1, -1)]
        elif randNum <= w + c1:
            parent2 = pLine
        else:
            parent2 = gLine
        start_pos = random.randint(0, len(parent1) - 1)
        end_pos = random.randint(0, len(parent1) - 1)
        if start_pos > end_pos: start_pos, end_pos = end_pos, start_pos
        croBird[start_pos:end_pos + 1] = parent1[start_pos:end_pos + 1].copy()
        list2 = list(range(0, start_pos))
        list1 = list(range(end_pos + 1, len(parent2)))
        list_index = list1 + list2
        j = -1
        for i in list_index:
            for j in range(j + 1, len(parent2) + 1):
                if parent2[j] not in croBird:
                    croBird[i] = parent2[j]
                    break
        return croBird

    def run(self):
        birdPop = [self.greedy(self.Customer, self.dis_matrix.copy()) for i in range(self.birdNum)]
        birdPop_car, fits = self.calFitness(birdPop, self.Demand, self.dis_matrix, self.CAPACITY, self.DISTABCE, self.C0, self.C1)
        gBest = pBest = min(fits)
        gLine = pLine = birdPop[fits.index(min(fits))]
        gLine_car = pLine_car = birdPop_car[fits.index(min(fits))]
        bestfit = [gBest]
        iterI = 1
        while iterI <= self.iterMax:
            for i in range(self.birdNum):
                birdPop[i] = self.crossover(birdPop[i], pLine, gLine, self.w, self.c1, self.c2)
            birdPop_car, fits = self.calFitness(birdPop, self.Demand, self.dis_matrix, self.CAPACITY, self.DISTABCE, self.C0, self.C1)
            pBest, pLine, pLine_car = min(fits), birdPop[fits.index(min(fits))], birdPop_car[fits.index(min(fits))]
            if min(fits) <= gBest:
                gBest, gLine, gLine_car = min(fits), birdPop[fits.index(min(fits))], birdPop_car[fits.index(min(fits))]
            bestfit.append(gBest)
            print(iterI, gBest)
            iterI += 1
        return gLine_car, gBest

    def draw_path(self, car_routes, CityCoordinates):
        for route in car_routes:
            x, y = [], []
            for i in route:
                Coordinate = CityCoordinates[i]
                x.append(Coordinate[0])
                y.append(Coordinate[1])
            x.append(x[0])
            y.append(y[0])
            # 绘制路径
            plt.plot(x, y, 'o-', alpha=0.8, linewidth=0.8)
            # 添加数字标签
            for idx, point in enumerate(route):
                plt.text(x[idx], y[idx], str(point), fontsize=8, ha='center', va='center')
            # 添加箭头表示方向
            for i in range(len(route)):
                plt.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]), 
                             arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('./path-ex3.pdf')
        plt.show()


class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PSO VRP Parameter Adjustment")
        self.pso = None

        # 设置字体
        self.font = tkFont.Font(family="DejaVu Sans", size=10)

        # 参数输入
        tk.Label(root, text="Particle Number:", font=self.font).grid(row=0, column=0)
        self.birdNum_entry = tk.Entry(root)
        self.birdNum_entry.insert(0, "50")
        self.birdNum_entry.grid(row=0, column=1)

        tk.Label(root, text="Inertia Factor w:", font=self.font).grid(row=1, column=0)
        self.w_entry = tk.Entry(root)
        self.w_entry.insert(0, "0.2")
        self.w_entry.grid(row=1, column=1)

        tk.Label(root, text="Cognitive Factor c1:", font=self.font).grid(row=2, column=0)
        self.c1_entry = tk.Entry(root)
        self.c1_entry.insert(0, "0.4")
        self.c1_entry.grid(row=2, column=1)

        tk.Label(root, text="Social Factor c2:", font=self.font).grid(row=3, column=0)
        self.c2_entry = tk.Entry(root)
        self.c2_entry.insert(0, "0.4")
        self.c2_entry.grid(row=3, column=1)

        tk.Label(root, text="Max Iterations:", font=self.font).grid(row=4, column=0)
        self.iterMax_entry = tk.Entry(root)
        self.iterMax_entry.insert(0, "50")
        self.iterMax_entry.grid(row=4, column=1)

        tk.Label(root, text="Vehicle Max Capacity:", font=self.font).grid(row=5, column=0)
        self.capacity_entry = tk.Entry(root)
        self.capacity_entry.insert(0, "120")
        self.capacity_entry.grid(row=5, column=1)

        tk.Label(root, text="Vehicle Max Distance:", font=self.font).grid(row=6, column=0)
        self.distance_entry = tk.Entry(root)
        self.distance_entry.insert(0, "250")
        self.distance_entry.grid(row=6, column=1)

        tk.Label(root, text="Vehicle Start Cost C0:", font=self.font).grid(row=7, column=0)
        self.c0_entry = tk.Entry(root)
        self.c0_entry.insert(0, "30")
        self.c0_entry.grid(row=7, column=1)

        tk.Label(root, text="Unit Distance Cost C1:", font=self.font).grid(row=8, column=0)
        self.c1_cost_entry = tk.Entry(root)
        self.c1_cost_entry.insert(0, "1")
        self.c1_cost_entry.grid(row=8, column=1)

        # 运行按钮
        self.run_button = tk.Button(root, text="Run PSO", font=self.font, command=self.run_pso)
        self.run_button.grid(row=9, column=0, columnspan=2)

        # 结果显示
        self.result_label = tk.Label(root, text="", font=self.font)
        self.result_label.grid(row=10, column=0, columnspan=2)

    def run_pso(self):
        try:
            birdNum = int(self.birdNum_entry.get())
            w = float(self.w_entry.get())
            c1 = float(self.c1_entry.get())
            c2 = float(self.c2_entry.get())
            iterMax = int(self.iterMax_entry.get())
            capacity = int(self.capacity_entry.get())
            distance = int(self.distance_entry.get())
            c0 = int(self.c0_entry.get())
            c1_cost = int(self.c1_cost_entry.get())

            self.pso = PSO_VRP(birdNum, w, c1, c2, iterMax, capacity, distance, c0, c1_cost)
            gLine_car, gBest = self.pso.run()
            self.result_label.config(text=f"Optimal Cost: {gBest}\nPath: {gLine_car}")
            self.pso.draw_path(gLine_car, self.pso.Customer)
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")


def main():
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()