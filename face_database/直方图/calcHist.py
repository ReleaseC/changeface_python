
'''
直方图均衡化：用它来提高图像的对比度
实例代码如下
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
 
img = cv2.imread('1.jpg', 0)
 
hist, bins = np.histogram(img.flatten(), 256, [0, 256])  # img.flatten()将数组变为一维数组
 
cdf = hist.cumsum()  # 计算直方图
cdf_normalized = cdf * hist.max() / cdf.max()
 
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
img2 = cdf[img]
 
cv2.imshow('original', img)
cv2.imshow('res', img2)
 
#plt.plot(cdf_normalized, color='b'), plt.hist(img.flatten(), 256, [0, 256], color='r'), plt.xlim([0, 256])
plt.plot(cdf, color='b'), plt.hist(img2.flatten(), 256, [0, 256], color='r'), plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()

