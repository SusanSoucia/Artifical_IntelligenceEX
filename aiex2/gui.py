import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from GA import GA
from otsu import OTSU

class GA_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("遗传算法图像二值化 GUI")

        # 设置中文字体
        self.chinese_font = ("SimSun", 10)

        # 参数输入
        self.frame_params = tk.Frame(root)
        self.frame_params.pack(pady=10)

        tk.Label(self.frame_params, text="种群大小 M:", font=self.chinese_font).grid(row=0, column=0)
        self.entry_M = tk.Entry(self.frame_params)
        self.entry_M.insert(0, "16")
        self.entry_M.grid(row=0, column=1)

        tk.Label(self.frame_params, text="选择概率 select_rate:", font=self.chinese_font).grid(row=1, column=0)
        self.entry_select_rate = tk.Entry(self.frame_params)
        self.entry_select_rate.insert(0, "0.5")
        self.entry_select_rate.grid(row=1, column=1)

        tk.Label(self.frame_params, text="精英比率 strong_rate:", font=self.chinese_font).grid(row=2, column=0)
        self.entry_strong_rate = tk.Entry(self.frame_params)
        self.entry_strong_rate.insert(0, "0.3")
        self.entry_strong_rate.grid(row=2, column=1)

        tk.Label(self.frame_params, text="变异概率 bianyi_rate:", font=self.chinese_font).grid(row=3, column=0)
        self.entry_bianyi_rate = tk.Entry(self.frame_params)
        self.entry_bianyi_rate.insert(0, "0.05")
        self.entry_bianyi_rate.grid(row=3, column=1)

        tk.Label(self.frame_params, text="迭代次数:", font=self.chinese_font).grid(row=4, column=0)
        self.entry_iterations = tk.Entry(self.frame_params)
        self.entry_iterations.insert(0, "10")
        self.entry_iterations.grid(row=4, column=1)

        tk.Label(self.frame_params, text="图像路径:", font=self.chinese_font).grid(row=5, column=0)
        self.entry_image_path = tk.Entry(self.frame_params, width=50)
        self.entry_image_path.insert(0, "./imgs/examples/2.jpg")
        self.entry_image_path.grid(row=5, column=1)
        tk.Button(self.frame_params, text="选择文件", font=self.chinese_font, command=self.select_image).grid(row=5, column=2)

        # 按钮
        self.btn_start = tk.Button(root, text="开始进化", font=self.chinese_font, command=self.start_evolution)
        self.btn_start.pack(pady=10)

        # 显示区域
        self.frame_display = tk.Frame(root)
        self.frame_display.pack()

        self.label_image = tk.Label(self.frame_display)
        self.label_image.pack()

        self.label_status = tk.Label(root, text="", font=self.chinese_font)
        self.label_status.pack()

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.bmp")])
        if file_path:
            self.entry_image_path.delete(0, tk.END)
            self.entry_image_path.insert(0, file_path)

    def start_evolution(self):
        try:
            M = int(self.entry_M.get())
            select_rate = float(self.entry_select_rate.get())
            strong_rate = float(self.entry_strong_rate.get())
            bianyi_rate = float(self.entry_bianyi_rate.get())
            iterations = int(self.entry_iterations.get())
            image_path = self.entry_image_path.get()

            # 读取图像
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                messagebox.showerror("错误", "无法读取图像")
                return

            # 初始化 GA
            ga = GA(gray, M)
            ga.select_rate = select_rate
            ga.strong_rate = strong_rate
            ga.bianyi_rate = bianyi_rate

            self.label_status.config(text="种群迭代中...")

            # 显示原图
            self.display_image(gray, "原图")

            # 进化过程
            for x in range(iterations):
                ga.evolution()
                if x % 2 == 0:  # 每2代显示一次中间结果
                    current_best = ga.get_threshold()
                    binary = self.threshold_image(gray, current_best)
                    self.display_image(binary, f"第 {x} 代最佳阈值: {current_best}")
                    self.root.update()  # 更新 GUI

            # 最终结果
            max_threshold = ga.get_threshold()
            result_image = self.threshold_image(gray, max_threshold)
            self.display_image(result_image, f"最终最佳阈值: {max_threshold}")
            self.label_status.config(text=f"完成！最佳阈值: {max_threshold}")

        except Exception as e:
            messagebox.showerror("错误", str(e))

    def threshold_image(self, image, threshold):
        ret, binary_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary_img

    def display_image(self, img, title):
        # 转换为 PIL Image
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.label_image.config(image=img_tk)
        self.label_image.image = img_tk
        self.label_status.config(text=title)

if __name__ == "__main__":
    root = tk.Tk()
    app = GA_GUI(root)
    root.mainloop()