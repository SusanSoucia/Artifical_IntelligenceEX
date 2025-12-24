import heapq
import time
import math
import matplotlib.pyplot as plt
import networkx as nx  # 使用networkx库来构建节点图
from collections import deque

# # 构建城市信息
# city_data = {
#     'Arad': [('Zerind', 75), ('Sibiu', 140), ('Timisoara', 118)],
#     'Zerind': [('Arad', 75), ('Oradea', 71)],
#     'Sibiu': [('Arad', 140), ('Oradea', 151), ('Fagaras', 99), ('Rimnicu', 80)],
#     'Timisoara': [('Arad', 118), ('Lugoj', 111)],
#     'Bucharest': [('Fagaras', 211), ('Pitesti', 101), ('Giurgiu', 90), ('Urziceni', 85)],
#     'Urziceni': [('Bucharest', 85), ('Hirsova', 98), ('Vaslui', 142)],
#     'Craiova': [('Drobeta', 120), ('Rimnicu', 146), ('Pitesti', 138)],
#     'Drobeta': [('Mehadia', 75), ('Craiova', 120)],
#     'Eforie': [('Hirsova', 86)],
#     'Fagaras': [('Sibiu', 99), ('Bucharest', 211)],
#     'Giurgiu': [('Bucharest', 90)],
#     'Hirsova': [('Urziceni', 98), ('Eforie', 86)],
#     'Iasi': [('Vaslui', 92), ('Neamt', 87)],
#     'Lugoj': [('Timisoara', 111), ('Mehadia', 70)],
#     'Mehadia': [('Lugoj', 70), ('Drobeta', 75)],
#     'Neamt': [('Iasi', 87)],
#     'Oradea': [('Zerind', 71), ('Sibiu', 151)],
#     'Pitesti': [('Rimnicu', 97), ('Bucharest', 101), ('Craiova', 138)],
#     'Rimnicu': [('Sibiu', 80), ('Pitesti', 97), ('Craiova', 146)],
#     'Sibiu': [('Rimnicu', 80), ('Fagaras', 99), ('Arad', 140), ('Oradea', 151)],
#     'Timisoara': [('Lugoj', 111), ('Arad', 118)],
#     'Urziceni': [('Vaslui', 142), ('Bucharest', 85), ('Hirsova', 98)],
#     'Vaslui': [('Iasi', 92), ('Urziceni', 142)],
#     'Zerind': [('Oradea', 71), ('Arad', 75)]}
#
# # 构建城市位置坐标
# city_coordinates = {
#     'Arad': (91, 492),
#     'Bucharest': (400, 327),
#     'Craiova': (253, 288),
#     'Drobeta': (165, 299),
#     'Eforie': (562, 293),
#     'Fagaras': (305, 449),
#     'Giurgiu': (375, 270),
#     'Hirsova': (534, 350),
#     'Iasi': (473, 506),
#     'Lugoj': (165, 379),
#     'Mehadia': (168, 339),
#     'Neamt': (406, 537),
#     'Oradea': (131, 571),
#     'Pitesti': (320, 368),
#     'Rimnicu': (233, 410),
#     'Sibiu': (207, 457),
#     'Timisoara': (94, 410),
#     'Urziceni': (456, 350),
#     'Vaslui': (509, 444),
#     'Zerind': (108, 531)}

# 新的城市连接与路径距离 (Edge Costs)
# 这里的距离略大于两点间的直线距离，模拟真实道路
city_data = {
    'Sector_A': [('Sector_B', 108), ('Sector_C', 145), ('Sector_D', 95)],
    'Sector_B': [('Sector_A', 108), ('Sector_E', 120), ('Sector_F', 85)],
    'Sector_C': [('Sector_A', 145), ('Sector_D', 70), ('Sector_G', 160)],
    'Sector_D': [('Sector_A', 95), ('Sector_C', 70), ('Sector_E', 80), ('Sector_H', 110)],
    'Sector_E': [('Sector_B', 120), ('Sector_D', 80), ('Sector_I', 130), ('Sector_F', 90)],
    'Sector_F': [('Sector_B', 85), ('Sector_E', 90), ('Sector_J', 100)],
    'Sector_G': [('Sector_C', 160), ('Sector_H', 75), ('Sector_K', 140)],
    'Sector_H': [('Sector_D', 110), ('Sector_G', 75), ('Sector_I', 65), ('Sector_L', 150)],
    'Sector_I': [('Sector_E', 130), ('Sector_H', 65), ('Sector_J', 80), ('Sector_M', 115)],
    'Sector_J': [('Sector_F', 100), ('Sector_I', 80), ('Sector_N', 125)],
    'Sector_K': [('Sector_G', 140), ('Sector_L', 95), ('Sector_Z', 210)],
    'Sector_L': [('Sector_H', 150), ('Sector_K', 95), ('Sector_M', 85), ('Sector_Z', 160)],
    'Sector_M': [('Sector_I', 115), ('Sector_L', 85), ('Sector_N', 70), ('Sector_Z', 130)],
    'Sector_N': [('Sector_J', 125), ('Sector_M', 70), ('Sector_Z', 180)],
    'Sector_Z': [('Sector_K', 210), ('Sector_L', 160), ('Sector_M', 130), ('Sector_N', 180)]
}

# 新的城市坐标 (Heuristics)
# 基于 600x600 的虚拟网格
city_coordinates = {
    'Sector_A': (50, 300),   # 起点附近
    'Sector_B': (120, 380),
    'Sector_C': (150, 200),
    'Sector_D': (130, 280),
    'Sector_E': (200, 320),
    'Sector_F': (200, 420),
    'Sector_G': (280, 180),
    'Sector_H': (240, 250),
    'Sector_I': (300, 330),
    'Sector_J': (310, 430),
    'Sector_K': (450, 150),
    'Sector_L': (400, 240),
    'Sector_M': (420, 340),
    'Sector_N': (430, 440),
    'Sector_Z': (550, 300)   # 终点附近
}
class Romania():
    def __init__(self):
        self.G = nx.Graph()
        self.coordinates = city_coordinates
        # 构建图
        for city, connections in city_data.items():
            self.G.add_node(city, pos=city_coordinates[city])
            for connection, weight in connections:
                self.G.add_edge(city, connection, weight=weight)

    def DFS(self, start, goal):
        runningTime = time.perf_counter()
        graph = self.G

        stack = [(start, [start])]
        nodes_expanded = []

        while stack:
            current, path = stack.pop()

            # 防止当前路径内形成环
            if current in path[:-1]:
                continue

            # 记录展开节点
            nodes_expanded.append(current)

            if current == goal:
                runningTime = time.perf_counter() - runningTime
                total_cost = sum(
                    graph.get_edge_data(path[i], path[i + 1])['weight']
                    for i in range(len(path) - 1)
                )
                return {
                    'path': path,
                    'nodes_expanded': nodes_expanded,
                    'total_cost': total_cost,
                    'time': runningTime
                }

            # 固定邻居顺序，保证 DFS 行为可复现
            for next_city in sorted(graph.neighbors(current), reverse=True):
                if next_city not in path:
                    stack.append((next_city, path + [next_city]))

        return None
    
    def BFS(self, start, goal):
        graph = self.G
        start_time = time.perf_counter()

        queue = deque([(start, [start])])
        visited = set([start])
        nodes_expanded = set()

        while queue:
            current, path = queue.popleft()
            nodes_expanded.add(current)

            if current == goal:
                total_cost = sum(
                    graph[path[i]][path[i + 1]]['weight']
                    for i in range(len(path) - 1)
                )
                running_time = time.perf_counter() - start_time
                return {
                    'path': path,
                    'nodes_expanded': nodes_expanded,
                    'total_cost': total_cost,
                    'time': running_time
                }

            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        running_time = time.perf_counter() - start_time
        return {
            'path': [],
            'nodes_expanded': nodes_expanded,
            'total_cost': float('inf'),
            'time': running_time
        }
    
    def A_star(self, start, goal):
        def heuristic(a, b):
            # 欧几里得距离（可采纳）
            return math.hypot(b[0] - a[0], b[1] - a[1])

        graph = self.G
        coords = self.coordinates

        start_time = time.perf_counter()

        # open list: (f, g, node, path)
        open_list = []
        heapq.heappush(open_list, (0, 0, start, [start]))

        # g-score 表：到每个节点的最小已知代价
        g_score = {start: 0}

        # 已扩展节点
        closed = set()

        nodes_expanded = set()

        while open_list:
            f, g, current, path = heapq.heappop(open_list)

            # 已处理过的节点跳过
            if current in closed:
                continue

            nodes_expanded.add(current)

            if current == goal:
                running_time = time.perf_counter() - start_time
                return {
                    'path': path,
                    'nodes_expanded': nodes_expanded,
                    'total_cost': g,
                    'time': running_time
                }

            closed.add(current)

            for neighbor in graph.neighbors(current):
                cost = graph[current][neighbor]['weight']
                tentative_g = g + cost

                # 如果发现更优路径
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(
                        coords[neighbor], coords[goal]
                    )
                    heapq.heappush(
                        open_list,
                        (f_score, tentative_g, neighbor, path + [neighbor])
                    )

        # 无解
        running_time = time.perf_counter() - start_time
        return {
            'path': [],
            'nodes_expanded': nodes_expanded,
            'total_cost': float('inf'),
            'time': running_time
        }
    
    def brute_force(self, start, goal):
        graph = self.G

        runningTime = time.perf_counter()

        leastCost = float('inf')
        leastPath = []

        nodes_expanded = set()

        def dfs(current, path, cost):
            nonlocal leastCost, leastPath

            nodes_expanded.add(current)

            # 到达目标
            if current == goal:
                if cost < leastCost:
                    leastCost = cost
                    leastPath = path.copy()
                return

            # 继续向下枚举
            for neighbor in graph.neighbors(current):
                if neighbor not in path:  # 防止环
                    edge_cost = graph[current][neighbor]['weight']
                    dfs(
                        neighbor,
                        path + [neighbor],
                        cost + edge_cost
                    )

        dfs(start, [start], 0)

        runningTime = time.perf_counter() - runningTime

        return {
            'path': leastPath,
            'nodes_expanded': nodes_expanded,
            'total_cost': leastCost,
            'time': runningTime
        }



# 广度优先搜索
def BFS(graph, start, goal):
    queue = [(start, [start])]
    nodes_expanded = []

    while queue:
        (current, path) = queue.pop(0)
        nodes_expanded.append(current)
        for next_city in set(graph.neighbors(current)) - set(path):
            if next_city == goal:
                total_cost = sum(graph.get_edge_data(path[i], path[i + 1])['weight'] for i in range(len(path) - 1))
                return {'path': path, 'nodes_expanded': nodes_expanded, 'total_cost': total_cost}
            else:
                queue.append((next_city, path + [next_city]))

    return None





# 深度优先搜索
def DFS(graph, start, goal):
    visited = set()
    stack = [(start, [start])]
    nodes_expanded = 0
    while stack:
        nodes_expanded += 1
        (current, path) = stack.pop()  # 使用栈，从栈顶移除元素
        if current not in visited:  # 检查当前节点是否已被访问
            visited.add(current)
            if current == goal:
                total_cost = sum(graph.get_edge_data(path[i], path[i + 1])['weight'] for i in range(len(path) - 1))
                return {'path': path, 'nodes_expanded': nodes_expanded, 'total_cost': total_cost}
            else:
                for next_city in graph.neighbors(current):  # 遍历所有邻居
                    if next_city not in visited:  # 只考虑未访问过的邻居
                        stack.append((next_city, path + [next_city]))  # 将邻居和路径压入栈

    return None  # 如果没有找到路径，则返回None


def heuristic(a, b):
    # 使用欧几里得距离作为启发式函数
    return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5


def A_star(graph, start, goal, coordinates):
    # 创建优先队列
    pq = []
    heapq.heappush(pq, (0, 0, start, [start]))  # (total_cost, new_cost, current, path)
    leastCost = float('inf')
    leastPath  = []
    visited = set()
    nodes_expanded = []
    while pq:
        _, current_weight, current, path = heapq.heappop(pq)
        nodes_expanded.append(current) 
        if current == goal:
            total_cost = sum(graph.get_edge_data(path[i], path[i + 1])['weight'] for i in range(len(path) - 1))
            if total_cost<= leastCost:
                leastCost = total_cost
                leastPath = path
            
        if current not in visited:
            visited = path
            for next_city in graph.neighbors(current):
                if next_city not in visited:
                    new_cost = current_weight + graph.get_edge_data(current, next_city)['weight']  # g(n)
                    heuristic_cost = heuristic(coordinates[next_city], coordinates[goal])  # h(n)
                    total_cost = new_cost + heuristic_cost
                    heapq.heappush(pq, (total_cost, new_cost, next_city, path + [next_city]))
    nodes_expanded.append(goal)
    nodes_expanded = set(nodes_expanded)
    return {'path': leastPath, 'nodes_expanded': nodes_expanded, 'total_cost': leastCost}


def main():
    # 创建图
    G = nx.Graph()

    # 添加城市节点和边
    for city, connections in city_data.items():
        G.add_node(city, pos=city_coordinates[city])
        for connection, weight in connections:
            G.add_edge(city, connection, weight=weight)

    # start, goal = 'Arad', 'Bucharest'
    start, goal = 'Sector_A', 'Sector_E'

    time1 = time.perf_counter()
    result = BFS(G, start, goal)
    time1 = time.perf_counter() - time1
    print("**********BFS**********")
    print("经过的节点数:", result['nodes_expanded'])
    print("路径:", result['path'] + [goal])
    print("总代价:", result['total_cost'])
    print("时间:", time1)
    print("")

    time1 = time.perf_counter()
    result = DFS(G, start, goal)
    time1 = time.perf_counter() - time1
    print("**********DFS**********")
    print("经过的节点数:", result['nodes_expanded'])
    print("路径:", result['path'] + [goal])
    print("总代价:", result['total_cost'])
    print("时间:", time1)
    print("")

    time1 = time.perf_counter()
    result = A_star(G, start, goal, city_coordinates)
    time1 = time.perf_counter() - time1
    print("**********A_star**********")
    print("经过的节点数:", result['nodes_expanded'])
    print("路径:", result['path'])
    print("总代价:", result['total_cost'])
    print("时间:", time1)

def vis():
    r = Romania()
    results = r.DFS("Sector_A","Sector_D")

    # 绘制
    fig = plt.figure(figsize=(7,6),dpi=100)
    ax= fig.add_subplot(111)
    pos = nx.spring_layout(r.G)
    nx.draw(
        r.G,
        pos,
        ax=ax,
        with_labels=True,
        node_size=1400,
        node_color="lightblue",
        edge_color="lightgray",
        font_size=10
    )
    edge_labels = nx.get_edge_attributes(r.G,"weight")
    nx.draw_networkx_edge_labels(r.G,pos,edge_labels=edge_labels,ax=ax)
    
    print(results['path'])

if __name__ == '__main__':
    vis()


