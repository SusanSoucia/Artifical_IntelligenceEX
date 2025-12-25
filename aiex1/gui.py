import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import font
from backends import Romania
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


root = tk.Tk()
root.title("Romania")
r = None


def confirm():
    srt = source.get()
    dst = destination.get()
    method = methods.get()
    global r
    
    def checkSame(srt,dst):
        if srt == dst:
            messagebox.showerror("TRY AGAIN","Srt cannot be the same as Dst!")
            return False
        if (not srt or not srt):
            messagebox.showerror("TRY AGAIN","Srt and Dst are needed")
            return False
        else:
            return True

    if checkSame(srt,dst):
        r = Romania()
        vis(r,srt,dst,method)


def vis(Romania,srt,dst,method ="BFS" ):
    gragh = Romania.G
    win = tk.Toplevel(root)
    win.title(f'This is {method}')

    fig = plt.figure(figsize=(7,6),dpi = 100)   
    ax= fig.add_subplot(111)

    if method=="DFS":
        results = Romania.DFS(srt,dst)
        path = results['path']

    if method =="BFS":
        results = Romania.BFS(srt,dst)
        path = results['path']

    if method =="A*":
        results =Romania.A_star(srt,dst)
        path = results['path']

    if method == "Brute_force":
        results = Romania.brute_force(srt,dst)
        path = results['path']

    pos = {city: city_coordinates[city] for city in gragh.nodes()}

    nx.draw(
        gragh,
        pos,
        ax=ax,
        with_labels=True,
        node_size=1200,
        node_color="lightblue",
        edge_color="lightgray",
        font_size=8
    )
    edge_labels = nx.get_edge_attributes(gragh, 'weight')

    if path and len(path) > 1:
        path_edges = list(zip(path[:-1], path[1:]))

        nx.draw_networkx_edges(
            gragh,
            pos,
            edgelist=path_edges,
            ax=ax,
            width=3,
            edge_color="red"
        )

        nx.draw_networkx_nodes(
            gragh,
            pos,
            nodelist=path,
            ax=ax,
            node_color="orange",
            node_size=1400
        )
    # 绘制其他信息
    info_text = (
        f"Algorithm: {method}\n"
        f"Path: {' -> '.join(path)}\n"
        f"Total cost: {results['total_cost']}\n"
        f"Expanded nodes: {len(results['nodes_expanded'])}\n"
        f"Time: {results['time']:.6f} s"
    )

    ax.text(
    0.00, 1.00,
    info_text,
    transform=ax.transAxes,
    ha='left',
    va='top',
    fontsize=9,
    bbox=dict(
        facecolor='white',
        alpha=0.5,     # 半透明
        edgecolor='gray',
        boxstyle='round,pad=0.3'
        )
    )
        # 最后绘制权重
    nx.draw_networkx_edge_labels(gragh, pos, edge_labels=edge_labels, ax=ax)

    win.canvas = FigureCanvasTkAgg(fig, master=win)
    win.canvas.draw()
    win.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    





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

values = list(city_coordinates.keys())
print(values)
source = ttk.Combobox(root)
source["values"] = values
source.pack()

destination = ttk.Combobox(root)
destination["values"] = values
destination.pack()

methods = ttk.Combobox(root)
methods["values"]=["DFS","BFS","A*","Brute_force"]
methods.pack()

button = ttk.Button(root,text="YES", command= confirm)
button.pack()


root.mainloop()