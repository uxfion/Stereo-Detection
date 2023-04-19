import cv2
import numpy as np
import time
import random
import math

# -----------------------------------双目相机的基本参数---------------------------------------------------------
#   left_camera_matrix          左相机的内参矩阵
#   right_camera_matrix         右相机的内参矩阵
#
#   left_distortion             左相机的畸变系数    格式(K1,K2,P1,P2,0)
#   right_distortion            右相机的畸变系数
# -------------------------------------------------------------------------------------------------------------
# # ----------- 默认参数 ----------------
# # 左镜头的内参，如焦距
# left_camera_matrix = np.array([[516.5066236,-1.444673028,320.2950423],[0,516.5816117,270.7881873],[0.,0.,1.]])
# right_camera_matrix = np.array([[511.8428182,1.295112628,317.310253],[0,513.0748795,269.5885026],[0.,0.,1.]])

# # 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
# left_distortion = np.array([[-0.046645194,0.077595167, 0.012476819,-0.000711358,0]])
# right_distortion = np.array([[-0.061588946,0.122384376,0.011081232,-0.000750439,0]])

# # 旋转矩阵
# R = np.array([[0.999911333,-0.004351508,0.012585312],
#               [0.004184066,0.999902792,0.013300386],
#               [-0.012641965,-0.013246549,0.999832341]])
# # 平移矩阵
# T = np.array([-120.3559901,-0.188953775,-0.662073075])

# # # ----------- 自己测的参数1 ----------------
# # 左镜头的内参，如焦距
# # 未转置
# left_camera_matrix = np.array([[518.5647686,-1.603209725,326.5599491],[0.,520.042716589812,251.574174043132],[0.,0.,1.]])
# right_camera_matrix = np.array([[516.145622534751,-0.623940185,318.4266868],[0.,518.032969390843,249.66554],[0.,0.,1.]])
# # # 转置
# # left_camera_matrix = np.array([[518.5647686,0.,0.],[-1.603209725,520.0427166,0.],[326.5599491,251.574174,1.]])
# # right_camera_matrix = np.array([[516.1456225,0.,0.],[-0.623940185,518.0329694,0.],[318.4266868,249.66554,1.]])

# # 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
# left_distortion = np.array([[-0.026301104,0.058693858,-0.00073206,-0.000131662,0]])
# right_distortion = np.array([[-0.031814632,0.087134826,-0.000793966,-0.001978622,0]])

# # 旋转矩阵
# R = np.array([[0.999996795,-0.000771654,-0.00241128],
#               [0.000768875,0.999999039,-0.001153414],
#               [0.002412167,0.001151556,0.999996428]])
# # 平移矩阵
# T = np.array([-119.8031438,-0.330010754,1.547692457])

# # ----------- 自己测的参数2 转置了 ----------------
# # 左镜头的内参，如焦距
# left_camera_matrix = np.array([[517.9892051,0.,0.],[-1.066133931,519.0762626,0.],[333.293805,242.5294729,1.]])
# right_camera_matrix = np.array([[515.302553,0.,0.],[-0.930908292,516.5674727,0.],[325.845992,241.4995264,1.]])

# # 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
# left_distortion = np.array([[-0.02894776,0.064523798,-0.005889258,0.003062348,0]])
# right_distortion = np.array([[-0.032744574,0.08277854,-0.005810594,0.001685287,0]])

# # 旋转矩阵
# R = np.array([[0.999991711,0.000180707,-0.00406766],
#               [-0.000191964,0.999996153,-0.002767233],
#               [0.004067144,0.002767991,0.999987898]])
# # 平移矩阵
# T = np.array([-119.5757932,-0.23286698,-0.363136026])

# ----------- 自己测的参数3(Taipa) ----------------
# 左镜头的内参，如焦距
left_camera_matrix = np.array([[551.2871943,-0.933422257,327.3979119],[0.,552.2345057,243.4281575],[0.,0.,1.]])
right_camera_matrix = np.array([[548.3658621,-0.157771511,322.9997923],[0.,548.6897439,244.8005207],[0.,0.,1.]])

# 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
left_distortion = np.array([[0.014205653,0.08308967,-0.002841747,0.000434416,0]])
right_distortion = np.array([[0.012185077,0.08856341,-0.00321925,0.001418914,0]])

# 旋转矩阵
R = np.array([[0.999997121,-0.000595908,-0.002324523],
              [0.000579543,0.999975091,-0.007034258],
              [0.002328657,0.007032891,0.999972558]])
# 平移矩阵
T = np.array([-132.9740431,-1.027962277,1.006308188])

size = (640, 480)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
print(Q)

# --------------------------鼠标回调函数---------------------------------------------------------
#   event               鼠标事件
#   param               输入参数
# -----------------------------------------------------------------------------------------------
def onmouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        print('\n像素坐标 x = %d, y = %d' % (x, y))
        # print("世界坐标是：", threeD[y][x][0], threeD[y][x][1], threeD[y][x][2], "mm")
        print("世界坐标xyz 是：", threeD[y][x][0] / 1000.0, threeD[y][x][1] / 1000.0, threeD[y][x][2] / 1000.0, "m")

        distance = math.sqrt(threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2)
        distance = distance / 1000.0  # mm -> m
        print("距离是：", distance, "m")


# 加载视频文件
capture = cv2.VideoCapture("./5.mp4")
# capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
WIN_NAME = 'Deep disp'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

# 读取视频
fps = 0.0
ret, frame = capture.read()
while ret:
    # 开始计时
    t1 = time.time()
    # 是否读取到了帧，读取到了则为True
    ret, frame = capture.read()
    # 切割为左右两张图片
    frame1 = frame[0:480, 0:640]
    frame2 = frame[0:480, 640:1280]
    # 将BGR格式转换成灰度图片，用于畸变矫正
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    # 转换为opencv的BGR格式
    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

    # ------------------------------------SGBM算法----------------------------------------------------------
    #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
    #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
    #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
    #                               取16、32、48、64等
    #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
    #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
    # ------------------------------------------------------------------------------------------------------
    blockSize = 3
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=1,
                                   numDisparities=64,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)
    # 计算视差
    disparity = stereo.compute(img1_rectified, img2_rectified)

    # 归一化函数算法，生成深度图（灰度图）
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 生成深度图（颜色图）
    dis_color = disparity
    dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dis_color = cv2.applyColorMap(dis_color, 2)

    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    # 计算出的threeD，需要乘以16，才等于现实中的距离
    threeD = threeD * 16

  

    #完成计时，计算帧率
    fps = (fps + (1. / (time.time() - t1))) / 2
    frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("depth", dis_color)
    cv2.imshow("left", frame1)
    cv2.imshow(WIN_NAME, disp)  # 显示深度图的双目画面

  # 鼠标回调事件
    cv2.setMouseCallback("depth", onmouse_pick_points, threeD)
    # cv2.setMouseCallback("left", onmouse_pick_points, threeD)
    # cv2.setMouseCallback("right", onmouse_pick_points, threeD)

    # 若键盘按下q则退出播放
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# 释放资源
capture.release()

# 关闭所有窗口
cv2.destroyAllWindows()
