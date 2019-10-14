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

def init_param():
    global url
    global user_img
    global openid
    global userjpg_src
    global user_txt
    global out_src
    global output_src
    global output1_src

    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()),dtype="uint8")
    user_img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    path = "/home/siiva/桌面/face_database/user/"+openid
    out_src = path +"/"+"out.mp4"

    if(not os.path.exists(path)):
        os.mkdir(path) 
    userjpg_src = "/home/siiva/桌面/face_database/user/"+openid+"/user.jpg"
    user_txt =  "/home/siiva/桌面/face_database/user/"+openid+"/user.txt"
    cv2.imwrite(userjpg_src,user_img)
    usertxt_dir = "/home/siiva/OpenFace/build/bin/FaceLandmarkImg -f " + userjpg_src + " -o " + user_txt
    print(usertxt_dir)
    os.system(usertxt_dir)
    command = "python3 /home/siiva/桌面/face_database/swap.py "+userjpg_src+" "+movie_src+" "+out_src+" "+user_txt+" "+actor_txt+" "+path+" "+audio_src+" "+openid
    print(command)
    os.system(command)


def receive_cmd(args): 
    global openid
    global movie_src
    global audio_src
    global url
    global actor_txt
    myjson=json.dumps(args)
    myjsondata = json.loads(myjson)
    if 'param' in myjsondata:
        if(myjsondata["param"]["action"] == "change_face"):
            openid = myjsondata["param"]["openid"]
            movie_src = "/home/siiva/桌面/face_database/video/video1/" + myjsondata["param"]["movie_id"]+".mp4"
            audio_src = "/home/siiva/桌面/face_database/video/video1/" + myjsondata["param"]["movie_id"]+".mp3"
            actor_list =  myjsondata["param"]["actor_ids"]
            url = myjsondata["param"]["url"]
            if(len(actor_list)>1):
                print("全换")
                actor_txt = "/home/siiva/桌面/face_database/video/video1/" + myjsondata["param"]["actor_ids"][0]+".txt"

            else:
                actor_txt = "/home/siiva/桌面/face_database/video/video1/" + myjsondata["param"]["actor_ids"][0]+".txt"
            actor_txt = "/home/siiva/桌面/face_database/video/video1/actor_all.txt"
            print("acotor_txt",actor_list)
            init_param()
           
def on_reconnect(args):
    print(args)

def on_disconnect(args):
    print(args)

def register_socket():
   
    socketIO = SocketIO('101.37.151.52', 3000, LoggingNamespace)
    try: 
        print("======register_socket======")
        getCodedata = {
                 "deviceId":"40B0765C676C"+"_python",
                 "type":"change_face",
        }
        socketIO.emit('cmd_register',getCodedata)
        socketIO.on('cmd', receive_cmd)
        socketIO.wait()
        
    except requests.exceptions.ConnectionError as e:
        print("[Warn]Please check your network with get_code_socket")
    except BaseException as e:
        print("[Warn]get_code_socket error",e) 



def change_face_socket():
    print("=====change_face_socket======")
    print(video_oss_path)
    uploaddata = {
        "param":{
            "action":"change_face_responds",
            "url":video_oss_path,
        },
        "from":"o35En4xXzxcDugU2QQ2ylp-jbX5k"
    }
    try:    
        with SocketIO('101.37.151.52', 3000, LoggingNamespace) as socketIO:
             socketIO.emit('cmd',uploaddata)
             
    except requests.exceptions.ConnectionError as e:
        print("[Warn]Please check your network")
        #check_and_restore_network(momeu_upload_socket())
    except BaseException as e:
        print("momeu_upload_socket error",e)

register_socket()
#oss()

