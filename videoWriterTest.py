# -*- coding=utf-8 -*-
import cv2  # 3.4.2
import os

video_path= 'static/upload/w.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # 视频帧率

# fourcc = cv2.CV_FOURCC('M', 'J', 'P', 'G')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videoWriter = cv2.VideoWriter('123.avi', fourcc, fps, (int(w), int(h)))  # (1280, 720)为视频大小

img_names = os.listdir("img")
# img_names.sort()

for i in range(0, len(img_names)):
    img_path = os.path.join("img", img_names[i])
    img = cv2.imread(img_path)
    print(img_path)
    # cv2.imshow('img', img12)
    # cv2.waitKey(1000/int(fps))
    videoWriter.write(img)
videoWriter.release()
print("finish\n")
