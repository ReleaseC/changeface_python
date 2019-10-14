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
import urllib.request
import face_recognition

def init_param():
    global errormsg
    global url
    global user_img
    global openid
    global userjpg_src
    global out_src
    global errormsg

    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()),dtype="uint8")
    user_img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    path = "/home/siiva/桌面/face_database/user/"+openid

    if(not os.path.exists(path)):
        os.mkdir(path) 
    #userjpg_src = "/home/siiva/桌面/face_database/user/"+openid+"/user_test.jpg"
    #cv2.imwrite(userjpg_src,user_img)
    
    result = has_face(user_img)
    if not result:
        #buhege
        errormsg = False
    else:
        #hege
        errormsg = True
    
    has_face_socket()
  
def has_face(frame):
    test_image = frame
    # Find faces in test image
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)
    print(face_encodings)
    #没有人
    if face_encodings == []:
        return False
    else:
        return True

def get_face_landmarks(user_photo, txtfile):
    global img_gray
    global landmarks_points, indexes_triangles
    global x1, y1, w1, h1
    global newvideo
    global errormsg
    global video_oss_path


    errormsg = ""
    file_txt = open(txtfile, 'r')
    cnt = 0
    s = file_txt.readline()  # 第一行是  frame=1

    landmarks = []
    landmarks_points = []
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
    face_points = [1, 2, 3 ,52,  15,
                   16, 17, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                   28, 29, 30, 31, 32, 33, 34, 35, 36]
    #face_points = [6,7,8,9,10,11,12]
    for idx, n in enumerate(face_points):
        landmarks_points.append(landmarks[n-1])

    img = cv2.imread(user_photo)  # 读取用户照片
    overlay = img.copy()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexhull, 255)
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

        #cv2.line(overlay, pt1, pt2, next_color, 1)
        #cv2.line(overlay, pt2, pt3, next_color, 1)
        #cv2.line(overlay, pt3, pt1, next_color, 1)
        # cv2.imshow("overlay",overlay)
        # cv2.waitKey(1)

    # cv2.waitKey(0)
    return img, img_face1


'''
    MAIN PROGRAM START
'''
def receive_cmd(args): 
    global openid
    global movie_src
    global audio_src
    global url
    global actor_txt
    
    myjson=json.dumps(args)
    myjsondata = json.loads(myjson)
    print(myjsondata)
    if 'param' in myjsondata:
        if(myjsondata["param"]["action"] == "has_face"):
            openid = myjsondata["param"]["openid"]        
            url = myjsondata["param"]["url"]
            init_param()
        if(myjsondata["param"]["action"] == "connect"):
            print("connect")
            #pass
            register_socket()

def dis_cmd():
    print("disconnect")   
    sys.exit()

def rec_cmd():
    print("reconnect") 

def on_reconnect(args):
    print(args)

def on_disconnect(args):
    print(args)


socketIO = SocketIO('101.37.151.52', 3000, LoggingNamespace)
def register_socket():
    try: 
        print("======register_socket======")
        getCodedata = {
                 "deviceId":"40B0765C676C"+"_check",
                 "type":"check_face",
        }
        socketIO.emit('cmd_register',getCodedata)
        socketIO.on('cmd', receive_cmd)
        socketIO.on('disconnect', dis_cmd)
        socketIO.on('reconnect', rec_cmd)
        socketIO.wait()
        
    except requests.exceptions.ConnectionError as e:
        print("[Warn]Please check your network with get_code_socket")
    #except BaseException as e:
    #    print("[Warn]get_code_socket error",e) 


    
def has_face_socket():
    print("=====change_face_socket======")
    print(errormsg)
    print(openid)
    uploaddata = {
        "param":{
            "action":"has_face_responds",
            "errormsg":errormsg,
        },
        "from":openid
    }
    try:    
        with SocketIO('101.37.151.52', 3000, LoggingNamespace) as socketIO:
             socketIO.emit('cmd',uploaddata)
             socketIO.on('disconnect', dis_cmd)
             
    except requests.exceptions.ConnectionError as e:
        print("[Warn]Please check your network")
        #check_and_restore_network(momeu_upload_socket())
    #except BaseException as e:
    #    print("momeu_upload_socket error",e)

register_socket()
print("bottom")
#oss()


