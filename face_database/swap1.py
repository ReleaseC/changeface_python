import cv2
import numpy as np
import dlib
import time
import sys
from color_transfer import color_transfer
import argparse
from PIL import Image
from socketIO_client_nexus import SocketIO, LoggingNamespace
import requests
import oss2
import json
import os
import ffmpeg
video_oss_path = ""

def change_face_socket():
    print("=====change_face_socket======")
    #print(video_oss_path)
    
    uploaddata = {
        "param":{
            "action":"change_face_responds",
            "url":video_oss_path,
            "rate":progress,
            "state":status,
            "movie":sys.argv[2].split(".")[0][-16:],
            "taskId":sys.argv[9]
        },
        "from":sys.argv[8]
    }

    print(uploaddata)
    try:    
        with SocketIO('101.37.151.52', 3000, LoggingNamespace) as socketIO:
             socketIO.emit('cmd',uploaddata)
             
    except requests.exceptions.ConnectionError as e:
        print("[Warn]Please check your network")
    except BaseException as e:
        print("momeu_upload_socket error",e)

########################
progress = 0.1
status = "change_start"
change_face_socket()
########################
def oss():
    global url
    global img_name
    global file_name
    global img_key
    global video_oss_path

    img_key = "1566898965st/"+sys.argv[9]+".mp4"
    img_path = videosrc+"/output.mp4"
    auth = oss2.Auth('LTAIAHvYCJg3q0sp', 'EOPCMRjjW3mDC8MFV4LSwAMEiMKVny')
    endpoint = 'https://oss-cn-hangzhou.aliyuncs.com'
    video_oss_path = "https://siiva-video-public.oss-cn-hangzhou.aliyuncs.com/" + img_key
    try:
        bucket = oss2.Bucket(auth, endpoint, 'siiva-video-public')
        bucket.put_object_from_file(img_key, img_path)
        print("upload success")
        url = video_oss_path
        change_face_socket()
    except Exception as e:
        print("upload err",e)

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def isFaceRoifun(landmarks_points, pt1, pt2, pt3):
    num = 0
    for item in landmarks_points:
        if item == pt2:
            num += 1
        if item == pt3:
            num += 1
        if item == pt1:
            num += 1
        else:
            pass
    if(num == 3):
        # print(1)
        return True
    else:
        num == 0
        return False


# 从landmarks数据里面找到下一个适配当前帧数的数据
def load_next_landmark(vfile, curr_frame):
    landmarks = []
    landmarks_points = []

    if vfile.closed:
        return None

    #print(' 读入下一组视频特征值坐标...')
    curr_file_pos = vfile.tell()  # 保存目前文件位置

    s = vfile.readline()  # 读下一行
    if not s:
        print('end of file')
        vfile.close()
        return None

    n = s.split('=')  # 应该是有 ‘FRAME=xxx\n’ 内容，分割为 FRAME xxx\n
    n1 = n[1].split('\n')     # 把 \n 去掉，留下 xxx 当前帧数

    if curr_frame != int(n1[0]):  # 目前画面是否有特征值
        print('没找到frame！')
        vfile.seek(curr_file_pos, 0)  # 回到上一行，之后再读才不会出错
        return None

    # 从文本文件里面读入所有68个特征值
    for i in range(68):
        s = vfile.readline()
        if not s:
            print('end of file')
            vfile.close()
            return None

        n = s.split(' ')  # 应该是两个 float 坐标 '123.44 343.44\n'
        n1 = n[1].split('\n')   # 去掉第二个数字的 \n
        x = int(float(n[0]))
        y = int(float(n1[0]))
        landmarks.append((x, y))  # 保存起来

    #face_points = [1, 2, 3,4, 52, 15,14,16, 17, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,28, 29, 30, 31, 32, 33, 34, 35, 36]
    face_points = [1,2,3,4,49,50,51,52,34,31,30,29,28,29,30,31,34,33,32,33,34,35,36,35,34,52,53,54,55,14,15,16,17,27,26,25,24,23,22,28,23,28,29,43,44,45,46,47,48,47,46,45,44,43,29,40,41,42,37,38,39,38,37,42,41,40,29,30,31,30,29,28,22,21,20,19,18,1]
   
    #face_points = [6,7,8,9,10,11,12]
    # 只处理嘴上的部分
    for idx, n in enumerate(face_points):
        landmarks_points.append(landmarks[n-1])

    return landmarks_points

# 从文本文件导入用户人脸特征值的数据


def get_face_landmarks(user_photo, txtfile):
    global img_gray
    global landmarks_points, indexes_triangles
    global x1, y1, w1, h1
    global newvideo
    global video_oss_path
    global state
    global landmarks_points_copy

    video_oss_path = []
    state = ""
    file_txt = open(txtfile, 'r')
    cnt = 0
    s = file_txt.readline()  # 第一行是  frame=1

    landmarks = []
    landmarks_points = []
    landmarks_points_copy = []
    #print(' 读入用户人脸特征值坐标...')
    while True:
        s = file_txt.readline()
        if not s:
            #print('end of file')
            file_txt.close()
            break
        n = s.split(' ')
        n1 = n[1].split('\n')
        #print(s, cnt, int(float(n[0])), int(float(n1[0])))
        x = int(float(n[0]))
        y = int(float(n1[0]))
        cnt += 1
        landmarks.append((x, y))

    # 只处理嘴上的部分
    '''face_points = [1, 2, 3 ,4,52, 15,14,
                   16, 17, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                   28, 29, 30, 31, 32, 33, 34, 35, 36]
    '''
    face_points = [1,2,3,4,49,50,51,52,34,31,30,29,28,29,30,31,34,33,32,33,34,35,36,35,34,52,53,54,55,14,15,16,17,27,26,25,24,23,22,28,23,28,29,43,44,45,46,47,48,47,46,45,44,43,29,40,41,42,37,38,39,38,37,42,41,40,29,30,31,30,29,28,22,21,20,19,18,1]
    #face_points = [1,2,3,32,33,34,35,36,15,16,17,27,26,25,24,23,22,21,20,19,18,1,37,38,39,28,43,44,45,46]
    for idx, n in enumerate(face_points):
        landmarks_points.append(landmarks[n-1])

    img = cv2.imread(user_photo)  # 读取用户照片
    overlay = img.copy()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    points = np.array(landmarks_points, np.int32)
    #points_copy = np.array(landmarks_points_copy, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexhull, 255)
    #cv2.fillPoly(mask,[convexhull],255)

    face_image_1 = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(convexhull)
    (x1, y1, w1, h1) = rect
    img_face1 = img[y1: y1 + h1, x1: x1 + w1]  # 输入照片的人脸区域

    # Delaunay triangulation
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)

    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    cnt = r = g = b = 0
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        result = isFaceRoifun(landmarks_points, pt1, pt2, pt3)

        index_pt1 = np.where((points == pt1).all(axis=1))  # 横向
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

        next_color = (0, 255, 0)
        cnt += 1

    return img, img_face1


'''
    MAIN PROGRAM START
'''

if len(sys.argv) != 10:
    print('usage: python ', sys.argv[0],
          'in.jpg in.mp4 out.mp4 in.txt out.txt path audio.mp3 openid taskid')
    exit(0)

# 全局变量，在 get_face_landmarks() 里面设定，给予之前初始值。
x1 = y1 = w1 = h1 = 0
indexes_triangles = []

# 打开个人图片，特征值坐标，设定它的三角型ARRAY
img, img_face1 = get_face_landmarks(sys.argv[1], sys.argv[4])
    
#sys.exit()
    
# 打开换脸视频
cap = cv2.VideoCapture(sys.argv[2])
if not cap.isOpened():
    print('无法打开视频文件', sys.argv[2])
    exit(0)

fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧数
ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 打开输出视频文件文件名,mp4 压缩格式。
fourcc = cv2.VideoWriter_fourcc(*'X264')
newvideo= sys.argv[6] +"/out1.mp4"
videoWriter1 = cv2.VideoWriter(newvideo, fourcc, fps, (ow, oh), True)
if not videoWriter1.isOpened():
    print('无法打开视频文件',newvideo)
    exit(0)

# 打开换脸视频的特征值文本文件TXT
video_in_file = open(sys.argv[5], 'r')

cnt = curr_frame = 0 


def change_face_main():
    global curr_frame
    global videosrc
    global status
    global progress
    

    progress = 0
    status = ""
    ########################
    progress = 0.2
    status = "changing"
    change_face_socket()
    ########################
    flag = True
    flagcount=0
    while True:
       # print(flagcount)
        flagcount=flagcount+1
        if(flagcount == 100 and flag):
            ########################
            progress = 0.3
            status = "changing"
            change_face_socket()
            ########################
        if(flagcount == 200 and flag):
            ########################
            progress = 0.4
            status = "changing"
            change_face_socket()
            ########################
        if(flagcount == 300 and flag):
            ########################
            progress = 0.5
            status = "changing"
            change_face_socket()
            ########################
        if(flagcount == 400 and flag):
            flag==False
            ########################
            progress = 0.6
            status = "changing"
            change_face_socket()
            ########################

        err, img2 = cap.read()  # 读入换脸视频下一帧
        if not err:
            break

        curr_frame += 1  # 记录目前帧数,第一帧从1开始

        # 看这帧是否有对应的 landmarks 数值
        landmarks_points2 = load_next_landmark(video_in_file, curr_frame)

        # 如果没有特征值就直接写出
        if landmarks_points2 == None or len(landmarks_points) == 0:
            print('没有特征值，直接输出', curr_frame)
            videoWriter1.write(img2)
            #cv2.imshow('mixed', img2)  # 合成视频
            #cv2.waitKey(1)
            continue

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2_new_face = np.zeros_like(img2)

        points2 = np.array(landmarks_points2, np.int32)
        points2_copy = np.array(landmarks_points2, np.int32)

        if len(points2) == 0:
            continue
        convexhull2 = cv2.convexHull(points2)

        rect3 = cv2.boundingRect(convexhull2)
        #print(len(points2), rect3)
        (x2, y2, w2, h2) = rect3
        img_face_out = img2[y2:y2 + h2, x2:x2 + w2]  # 替换视频内的人脸区域
        # 两张脸做颜色处理。新脸的颜色色调必须配合视频内的脸的颜色色调。
        #img_face2 = color_transfer(img_face_out, img_face1)

        img_face2 = img_face1
        img[y1: y1 + h1, x1: x1 + w1] = img_face1
        lines_space_mask = np.zeros_like(img_gray)
        lines_space_new_face = np.zeros_like(img2)

        
        # Triangulation of both faces
        for triangle_index in indexes_triangles:
            # Triangulation of the first face
            tr1_pt1 = landmarks_points[triangle_index[0]]
            tr1_pt2 = landmarks_points[triangle_index[1]]
            tr1_pt3 = landmarks_points[triangle_index[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

            rect1 = cv2.boundingRect(triangle1)
            (x, y, w, h) = rect1
            cropped_triangle = img[y: y + h, x: x + w]

            cropped_tr1_mask = np.zeros((h, w), np.uint8)

            points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                            [tr1_pt2[0] - x, tr1_pt2[1] - y],
                            [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
            #cv2.fillPoly(cropped_tr1_mask, [points],255)
            # Triangulation of second face
            tr2_pt1 = landmarks_points2[triangle_index[0]]
            tr2_pt2 = landmarks_points2[triangle_index[1]]
            tr2_pt3 = landmarks_points2[triangle_index[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

            rect2 = cv2.boundingRect(triangle2)
            (x, y, w, h) = rect2

            cropped_tr2_mask = np.zeros((h, w), np.uint8)
            points2 = np.array([[tr2_pt1[0] - x - 1, tr2_pt1[1] - y - 1],
                                [tr2_pt2[0] - x - 1, tr2_pt2[1] - y - 1],
                                [tr2_pt3[0] - x - 1, tr2_pt3[1] - y - 1]], np.int32)

            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
            
            #cv2.fillPoly(cropped_tr2_mask, [points2_copy], 255)
            # Warp triangles
            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)

            # 需要去除黑边，使用 CV2.INTER_NEARESTs
            warped_triangle = cv2.warpAffine(
                cropped_triangle, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
            # cv2.warpAffine(cropped_triangle, M, (w, h),
            #               warped_triangle, cv2.INTER_NEAREST)
            warped_triangle = cv2.bitwise_and(
                warped_triangle, warped_triangle, mask=cropped_tr2_mask)

            # Reconstructing destination face
            
            img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
            img2_new_face_rect_area_gray = cv2.cvtColor(
                img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
            _, mask_triangles_designed = cv2.threshold(
                img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(
                warped_triangle, warped_triangle, mask=mask_triangles_designed)

            img2_new_face_rect_area = cv2.add(
                img2_new_face_rect_area, warped_triangle)
            
            img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

       
        # Face swapped (putting 1st face into 2nd face) (233,214,207)
        img2_face_mask = np.zeros_like(img2_gray)
        img2_head_mask = cv2.fillPoly(img2_face_mask, [points2_copy], (255))
        #img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, (255))
        img2_face_mask = cv2.bitwise_not(img2_head_mask)
        #cv2.imshow("img2_head_mask",img2_head_mask)

        img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
        #cv2.imshow("img1",img2_head_noface)
        #cv2.imshow("mask",img2_face_mask)
        #############################################
        #imgbg = cv2.imread("/home/siiva/桌面/face_database/bg.jpg")
        #imgbg = cv2.resize(imgbg, (img2_head_noface.shape[1], img2_head_noface.shape[0]))

        #Mask = img2_face_mask
        #Mask_inv = cv2.bitwise_not(Mask)

        #腐蚀 Mask_inv
        #kernel = np.ones((5, 5), np.uint8)
        #erosion = cv2.erode(Mask_inv, kernel)  # 腐蚀
        #膨胀
        #dilation = cv2.dilate(Mask_inv, kernel, iterations = 5)
        #erosion_mask = cv2.bitwise_not(dilation)
        

        #kernel = np.ones((30, 30), np.uint8)
        #img2_face_mask =  cv2.erode(img2_face_mask, kernel)  # 腐蚀
        #cv2.imshow("img2_face_mask", img2_face_mask)
        #pifu_mask = cv2.bitwise_and(imgbg, imgbg, mask=Mask_inv)
        #res = cv2.add(img2_head_noface, pifu_mask)
        #result = cv2.bitwise_and(res, res, mask=erosion_mask)

        #img2_head_noface = result
        
        #########################################
        result = cv2.add(img2_head_noface, img2_new_face)
        
        (x, y, w, h) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
        seamlessclone = cv2.seamlessClone(
            result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

        #cv2.imshow('mixed', seamlessclone)  # 合成视频
        videoWriter1.write(seamlessclone)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            break

        if key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
    videoWriter1.release()
    
    ########################
    progress = 0.8
    status = "changing"
    change_face_socket()
    ########################
     
    videosrc = sys.argv[6]
    audio_src = sys.argv[7]
    #time.sleep(3)
    #command1 = "sudo ffmpeg -i "+videosrc+"/out.mp4 -vcodec h264 -y "+videosrc+"/output.mp4"
    #print(command1)
    #os.system(command1)

    command2 = "ffmpeg -i "+videosrc+"/out1.mp4"+" -i "+audio_src+" -y -c:v copy -c:a aac -strict experimental "+videosrc+"/output.mp4"
    print(command2)
    os.system(command2)
    oss()
    ########################
    progress = 1
    status = "changed"
    change_face_socket()
    ending = True
    ########################
    sys.exit(0)

change_face_main()



