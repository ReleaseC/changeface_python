import cv2
import numpy as np
imgyuan = cv2.imread("/home/siiva/桌面/face_database/img1.jpg")
mask = cv2.imread("/home/siiva/桌面/face_database/mask.jpg")

imgbg = cv2.imread("/home/siiva/桌面/face_database/bg.jpg")
imgbg = cv2.resize(imgbg, (imgyuan.shape[1], imgyuan.shape[0]))

Mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
Mask_inv = cv2.bitwise_not(Mask)

#腐蚀 Mask_inv
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(Mask_inv, kernel)  # 腐蚀
erosion_mask = cv2.bitwise_not(erosion)

pifu_mask = cv2.bitwise_and(imgbg, imgbg, mask=Mask_inv)
res = cv2.add(imgyuan, pifu_mask)
result = cv2.bitwise_and(res, res, mask=erosion_mask)
cv2.imwrite("result.jpg",result)

