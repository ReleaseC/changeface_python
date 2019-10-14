#!/usr/bin/env python
#-*- coding:utf8 -*-
#Author: hanli
#Update: 2018-09-24

from __future__ import print_function

import os
import sys
import subprocess
from ffprobe3 import FFProbe
from ffprobe3.exceptions import FFProbeError

class MainFun():

  '''
  color
  '''
  def __init__(self):

    self.left = '\033[1;31;40m'
    self.gren = '\033[1;32;40m'
    self.right = '\033[0m'
    self.videos = sys.argv[1]

  def CheckModule(self):

    try:
      from ffprobe3 import FFProbe
      from ffprobe3.exceptions import FFProbeError
    except:
      self.ConsoleLog("Not install ffprobe3, please do 'pip install ffprobe3'","warn")

    return True

  '''
  checkvideo valid
  '''
  def CheckVideo(self):

    try:

      media = FFProbe(self.videos)

      for index, stream in enumerate(media.streams, 1):
        self.ConsoleLog('\tStream: {0}'.format(index))

        try:
            if stream.is_video():
                frame_rate = stream.frames() / stream.duration_seconds()
                self.ConsoleLog('\t\tFrame Rate:{0}'.format(frame_rate))
                self.ConsoleLog('\t\tFrame Size:{0}'.format(stream.frame_size()))
            self.ConsoleLog('\t\tDuration:{0}'.format(stream.duration_seconds()))
            self.ConsoleLog('\t\tFrames:{0}'.format(stream.frames()))

            if stream.is_video():
              self.ConsoleLog('\t\tIs video: True')
              self.ConsoleLog('\t\tvideo encode:{0}'.format(stream.codec()))
              self.CheckResult(stream.codec(),"video")

            if stream.is_audio():
              self.ConsoleLog('\t\tIs audio: True')
              self.ConsoleLog('\t\taudio encode:{0}'.format(stream.codec()))
              self.CheckResult(stream.codec(),"audio")

        except FFProbeError as e:
            self.ConsoleLog(e,"warn")
        except Exception as e:
            self.ConsoleLog(e,"warn")

      return True

    except Exception as e:
      self.ConsoleLog(e,"warn")

  '''
  check result
  '''
  def CheckResult(self,codec,types=None):

    if types == 'video':
      if codec.lower() in ['h264','h265','h263','vp9','vp8','theora']:
        self.ConsoleLog("Chrom can playing video","result")
      elif codec.lower() in ['h264','theora']:
        self.ConsoleLog("FireFox and Safari can playing video","result")
      else:
        self.ConsoleLog("Chrom and FireFox not Support video encode type {0}".format(codec),"exec")

    if types == 'audio':
      if codec.lower() in ['vorbis','wmv','aac','mp3']:
        self.ConsoleLog("Chrom and FireFox can playing audio","result")
      else:
        self.ConsoleLog("Chrom FireFox and Safari not Support audio encode type {0}".format(codec),"exec")

  '''
  output log
  '''
  def ConsoleLog(self,level,tag=None):

    if tag == "warn":
      sys.exit("{0}[ERROR:]{1}{2}".format(self.left,self.right,level))
    elif tag == "result":
      print("{0}[CheckResult:]{1}{2}".format(self.gren,self.right,level))
    if tag == "exce":
      print("{0}[ERROR:]{1}{2}".format(self.left,self.right,level))
    else:
      print("[INFO:]{2}".format(self.gren,self.right,level))

'''
Main input
'''
if __name__ == '__main__':


  if MainFun().CheckModule():
    if MainFun().CheckVideo() == None:
      sys.exit('\033[1;31;40m[ERROR:]\033[0mInput file is not video file')
