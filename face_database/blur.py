
# encoding=utf8
 
from PIL import Image
import cv2
import sys
import os

def sum_9_region_new(img, x, y):
	'''确定噪点 '''
	cur_pixel = img.getpixel((x, y))  # 当前像素点的值
	width = img.width
	height = img.height
 
	if cur_pixel == 1:  # 如果当前点为白色区域,则不统计邻域值
		return 0
 
	# 因当前图片的四周都有黑点，所以周围的黑点可以去除
	if y < 3:  # 本例中，前两行的黑点都可以去除
		return 1
	elif y > height - 3:  # 最下面两行
		return 1
	else:  # y不在边界
		if x < 3:  # 前两列
			return 1
		elif x == width - 1:  # 右边非顶点
			return 1
		else:  # 具备9领域条件的
			sum = img.getpixel((x - 1, y - 1)) \
				  + img.getpixel((x - 1, y)) \
				  + img.getpixel((x - 1, y + 1)) \
				  + img.getpixel((x, y - 1)) \
				  + cur_pixel \
				  + img.getpixel((x, y + 1)) \
				  + img.getpixel((x + 1, y - 1)) \
				  + img.getpixel((x + 1, y)) \
				  + img.getpixel((x + 1, y + 1))
			return 9 - sum
 
def collect_noise_point(img):
	'''收集所有的噪点'''
	noise_point_list = []
	for x in range(img.width):
		for y in range(img.height):
			res_9 = sum_9_region_new(img, x, y)
			if (0 < res_9 < 3) and img.getpixel((x, y)) == 0:  # 找到孤立点
				pos = (x, y)
				noise_point_list.append(pos)
	return noise_point_list
 
def remove_noise_pixel(img, noise_point_list):
	'''根据噪点的位置信息，消除二值图片的黑点噪声'''
	for item in noise_point_list:
		img.putpixel((item[0], item[1]), 1)
 
def get_bin_table(threshold=115):
	'''获取灰度转二值的映射table,0表示黑色,1表示白色'''
	table = []
	for i in range(256):
		if i < threshold:
			table.append(0)
		else:
			table.append(1)
	return table
 
def main():
	if len(sys.argv)!=2:
		print("usage: python3 name.py test.jpg")
		return
	userimg = sys.argv[1]
	outimg = sys.argv[1]

	image = Image.open(userimg)
	imgry = image.convert('L')
	table = get_bin_table()
	binary = imgry.point(table, '1')
	noise_point_list = collect_noise_point(binary)
	remove_noise_pixel(binary, noise_point_list)
	binary.save('temp.png')

	img1 = cv2.imread(userimg)
	img3 = cv2.bilateralFilter(img1, 0, 60, 10)
	img3 = cv2.resize(img3, (img1.shape[1],img1.shape[0]))

	if os.path.exists("temp.png"):
		#生成mask
		img2 = cv2.imread('temp.png') #read treshold img
		img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
		Mask = img2gray
		Mask_inv = cv2.bitwise_not(img2gray)

		img1_bg = cv2.bitwise_and(img3, img3, mask=Mask)
		img1_fg = cv2.bitwise_and(img1, img1, mask=Mask_inv)

		dst = cv2.add(img1_bg, img1_fg)
		cv2.imwrite(outimg, dst)
	else:
		print("生成temp文件失败")
 
if __name__ == '__main__':
	main()
