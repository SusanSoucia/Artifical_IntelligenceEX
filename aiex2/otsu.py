import numpy as np


class OTSU:  # 定义大津算法类
    def otsu(self, image, threshold):  # 计算适应度 (类间方差)
        # 扁平化处理方便计算，虽然不扁平化也可以，但在做mask时扁平化逻辑更直观
        img_flat = image.ravel() 
        size = img_flat.size
        
        # bin_image_mask:前景为True，背景为False
        bin_image_mask = img_flat < threshold
        
        w0 = np.sum(bin_image_mask) # 前景像素数
        w1 = size - w0             # 背景像素数
        
        # 避免分母为0的情况
        if w0 == 0 or w1 == 0:
            return 0
        
        # 计算像素值总和
        sum_all = np.sum(img_flat)
        sum0 = np.sum(img_flat[bin_image_mask])
        sum1 = sum_all - sum0
        
        mean0 = sum0 / w0
        mean1 = sum1 / w1
        
        # 类间方差公式
        fitt = (w0 / size) * (w1 / size) * ((mean0 - mean1) ** 2)
        return fitt