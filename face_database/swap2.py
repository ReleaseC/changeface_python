import os
import sys
import ffmpeg
videosrc = "/home/siiva/桌面/face_database/user/o35En4zyGPcIysiq_inIxVxGC17M"
#audio_src = sys.argv[2]
def ffmpeg_recode():
    input_file = videosrc+"/out.mp4"
    output_file = videosrc+"/output.mp4"
    out, err = (
        ffmpeg
            .input(input_file)
            .output(output_file, '-vcodec h264')
            .run(overwrite_output=True)
    )
    if out == b'':
        print('do nothing')

ffmpeg_recode()