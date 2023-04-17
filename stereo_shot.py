import cv2
import time
import os

counter = 1
AUTO = True  # 自动拍照，或手动按s键拍照
INTERVAL = 2 # 自动拍照间隔
camera = cv2.VideoCapture(1)#也许你可能要capture两次
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)#设置分辨率
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)#
utc = time.time()
folder = "./data/data_3" # 拍照文件目录
if not os.path.exists(folder):
    os.mkdir(folder)
    os.mkdir(folder + "/left")
    os.mkdir(folder + "/right")

def shot( frame):
    global counter
    leftpath = folder +"/left/"+"left_" + str(counter) + ".jpg"
    rightpath=folder + "/right/"+ "right_" + str(counter) + ".jpg"
    leftframe=frame[0:480,0:640]#这里是为了将合在一个窗口显示的图像分为左右摄像头
    rightframe=frame[0:480,640:1280]
    cv2.imwrite(leftpath, leftframe)

    cv2.imwrite(rightpath, rightframe)
    print("snapshot saved into: " + leftpath)
    print("snapshot saved into: " + rightpath)

while True:
    ret, frame = camera.read()

    cv2.imshow("original", frame)

    now = time.time()
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        shot( frame)
        counter += 1

camera.release()
cv2.destroyWindow("original")
