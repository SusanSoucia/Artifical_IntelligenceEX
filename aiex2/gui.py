# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import locale
import time
from GA import GA
from otsu import OTSU

# 设置 locale 为中文
try:
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
except locale.Error:
    pass  # 如果不支持，跳过

class GA_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GA Image Binarization GUI")
        self.root.geometry("600x700")  # 设置窗口大小

        # 设置中文字体
        self.chinese_font = ("WenQuanYi Zen Hei", 10)

        # 参数输入
        self.frame_params = tk.Frame(root)
        self.frame_params.pack(pady=10)

        tk.Label(self.frame_params, text="Population Size M:", font=self.chinese_font).grid(row=0, column=0)
        self.entry_M = tk.Entry(self.frame_params)
        self.entry_M.insert(0, "16")
        self.entry_M.grid(row=0, column=1)

        tk.Label(self.frame_params, text="Selection Rate:", font=self.chinese_font).grid(row=1, column=0)
        self.entry_select_rate = tk.Entry(self.frame_params)
        self.entry_select_rate.insert(0, "0.5")
        self.entry_select_rate.grid(row=1, column=1)

        tk.Label(self.frame_params, text="Elite Rate:", font=self.chinese_font).grid(row=2, column=0)
        self.entry_strong_rate = tk.Entry(self.frame_params)
        self.entry_strong_rate.insert(0, "0.3")
        self.entry_strong_rate.grid(row=2, column=1)

        tk.Label(self.frame_params, text="Mutation Rate:", font=self.chinese_font).grid(row=3, column=0)
        self.entry_bianyi_rate = tk.Entry(self.frame_params)
        self.entry_bianyi_rate.insert(0, "0.05")
        self.entry_bianyi_rate.grid(row=3, column=1)

        tk.Label(self.frame_params, text="Iterations:", font=self.chinese_font).grid(row=4, column=0)
        self.entry_iterations = tk.Entry(self.frame_params)
        self.entry_iterations.insert(0, "10")
        self.entry_iterations.grid(row=4, column=1)

        tk.Label(self.frame_params, text="Image Path:", font=self.chinese_font).grid(row=5, column=0)
        self.entry_image_path = tk.Entry(self.frame_params, width=50)
        self.entry_image_path.insert(0, "./imgs/examples/2.jpg")
        self.entry_image_path.grid(row=5, column=1)
        tk.Button(self.frame_params, text="Select File", font=self.chinese_font, command=self.select_image).grid(row=5, column=2)

        # 按钮
        self.btn_start = tk.Button(root, text="Start Evolution", font=self.chinese_font, command=self.start_evolution)
        self.btn_start.pack(pady=10)

        # 显示区域
        self.frame_display = tk.Frame(root)
        self.frame_display.pack()

        self.label_image = tk.Label(self.frame_display)
        self.label_image.pack()

        self.btn_save = tk.Button(self.frame_display, text="Save Result", command=self.save_result, state=tk.DISABLED)
        self.btn_save.pack(pady=5)

        self.result_image = None
        self.max_threshold = None

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

            self.label_status.config(text="Evolution in progress...")

            # 显示原图
            self.display_image(gray, "Original Image")

            # 进化过程（不显示中间结果）
            for x in range(iterations):
                ga.evolution()

            # 最终结果
            max_threshold = ga.get_threshold()
            result_image = self.threshold_image(gray, max_threshold)
            self.result_image = result_image
            self.max_threshold = max_threshold
            self.display_image(result_image, f"Final Best Threshold: {max_threshold}")
            self.label_status.config(text=f"Completed! Best Threshold: {max_threshold}")
            self.btn_save.config(state=tk.NORMAL)  # 启用保存按钮
            messagebox.showinfo("Result", f"The best threshold found is: {max_threshold}")

        except Exception as e:
            messagebox.showerror("错误", str(e))

    def threshold_image(self, image, threshold):
        ret, binary_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary_img

    def save_result(self):
        if self.result_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
            if file_path:
                cv2.imwrite(file_path, self.result_image)
                messagebox.showinfo("Saved", f"Result saved to {file_path}")
        else:
            messagebox.showerror("Error", "No result to save")

    def display_image(self, img, title):
        # 缩放图像到合适大小（例如 400x300）
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((400, 300), Image.LANCZOS)  # 缩放
        img_tk = ImageTk.PhotoImage(img_pil)
        self.label_image.config(image=img_tk)
        self.label_image.image = img_tk
        self.label_status.config(text=title)

if __name__ == "__main__":
    root = tk.Tk()
    app = GA_GUI(root)
    root.mainloop()