import torch

import cv2
import argparse
import os
def parse_args(videopath,framesave):
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Process pic')
    parser.add_argument('--input', help='video to process', dest='input', default=None, type=str)
    parser.add_argument('--output', help='pic to store', dest='output', default=None, type=str)
    #input为输入视频的路径 ，output为输出存放图片的路径
    args = parser.parse_args(['--input',videopath,'--output',framesave])
    return args

def process_video(i_video, o_video):
    cap = cv2.VideoCapture(i_video)
    # VideoCapture()中的参数若为0，则表示打开笔记本的内置摄像头
    # 若为视频文件路径，则表示打开视频

    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) # 获取视频总帧数
    # print(num_frame)

    expand_name = '.jpg'
    if not cap.isOpened():
        print("Please check the path.")

    cnt = 0
    while 1:
        ret, frame = cap.read()
        # cap.read()表示按帧读取视频。ret和frame是获取cap.read()方法的两个返回值
        # 其中，ret是布尔值。如果读取正确，则返回TRUE；如果文件读取到视频最后一帧的下一帧，则返回False
        # frame就是每一帧的图像

        if not ret:
            break

        cnt += 1 # 从1开始计帧数
        cv2.imwrite(os.path.join(o_video, str(cnt) + expand_name), frame)

if __name__ == '__main__':
    videopath=r"input/video/test.mp4"
    framesave=r"videoframe/frame3"
    args = parse_args(videopath,framesave)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print('Called with args:')
    print(args)
    process_video(args.input, args.output)

