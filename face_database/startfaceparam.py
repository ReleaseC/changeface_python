import time
import requests
import json
import urllib.request
import numpy as np
import cv2
import os
import sys
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
    os.system("python3 /home/siiva/桌面/face_database/blur.py "+userjpg_src)

    usertxt_dir = "/home/siiva/OpenFace/build/bin/FaceLandmarkImg -f " + userjpg_src + " -o " + user_txt
    #print(usertxt_dir)
    os.system(usertxt_dir)
    command = "python3 /home/siiva/桌面/face_database/swap.py "+userjpg_src+" "+movie_src+" "+out_src+" "+user_txt+" "+actor_txt+" "+path+" "+audio_src+" "+openid+" "+taskid
    os.system(command)
    if(time.time()-starttime>10800):
        print(time.time()-starttime)
        print("exit")
        sys.exit()

def get_actor(myjsondata):
    movieid = movie_src.split(".")[0][-16:]
    #print(movieid)
    txt_str = "actor_all.txt"
    #if len(actor_list) == 3:
    #    txt_str = "actor_all.txt"
    #elif len(actor_list) == 1:
    #    txt_str = actor_list[0] + ".txt"
    #elif len(actor_list) == 2:
    #    txt_str = actor_list[0]+"_"+actor_list[1]+".txt"
    return home_path+myjsondata["param"]["movie_id"]+"/" + txt_str
    
def receive_cmd(myjsondata): 
    print(myjsondata)
    global openid
    global movie_src
    global audio_src
    global url
    global actor_txt
    global taskid
    global actor_list
   # global myjsondata
    global home_path
    #myjson=json.dumps(args)
    #myjsondata = json.loads(myjson)
    home_path = "/home/siiva/桌面/face_database/video/"
    if(myjsondata["code"] == 0):
        if 'param' in myjsondata:
            #if(myjsondata["param"]["action"] == "change_face"):
            openid = myjsondata["param"]["openid"]
            taskid = myjsondata["param"]["taskId"]
            movie_src = home_path+myjsondata["param"]["movie_id"]+"/" + myjsondata["param"]["movie_id"]+".mp4"
            audio_src = home_path+myjsondata["param"]["movie_id"]+"/" + myjsondata["param"]["movie_id"]+".mp3"
            actor_list =  myjsondata["param"]["actor_ids"]
            url = myjsondata["param"]["file_name"]
            actor_txt = get_actor(myjsondata)
            #print("acotor_txt========",actor_list)
            init_param()
    else:
        print("none task")

def update_device():
    data_info = dict(device_id='40B0765C676C',state='free',type='change_face')
    update_res = requests.post("https://iva.siiva.com/me_photo/device/update?", json=data_info )
    update_data = update_res.text
    mydata = json.loads(update_data)
    if(mydata["code"] == 0):
        print("更新成功")
    else:
        print("更新失败")

update_device()
starttime= time.time()
while(True):
    time.sleep(1)
    #try:
    res_device = requests.get("https://iva.siiva.com/me_photo/device/info?device_id=40B0765C676C")
    data_device = res_device.text
    if data_device:
        data_myjson = json.loads(data_device)
        #print(data_device)
        if(data_myjson["state"]=="free"):
            res = requests.get("https://iva.siiva.com/me_photo/change_face/task")
            data = res.text
            myjson = json.loads(res.text)
            receive_cmd(myjson)
        else:
            pass
    
        
    