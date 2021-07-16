import cv2
import numpy as np
import os
import glob

CHECKERBOARD = (7,7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = []
# Создание вектора для хранения векторов 2D точек для каждого изображения шахматной доски
imgpoints = [] 
# Определение мировых координат для 3D точек
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Найти углы шахматной доски
    # Если на изображении найдено нужное количество углов, тогда ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    """
    Если желаемый номер угла обнаружен,
    уточняем координаты пикселей и отображаем
    их на изображениях шахматной доски
    """
    if ret == True:
        objpoints.append(objp)
        # уточнение координат пикселей для заданных 2d точек.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        
        imgpoints.append(corners2)
        # Нарисовать и отобразить углы
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.imshow("camera", img)
    if cv2.waitKey(10) == 27: # Клавиша Esc
        break
cap.release()
cv2.destroyAllWindows()

"""
Выполнение калибровки камеры с помощью
Передача значения известных трехмерных точек (объектов)
и соответствующие пиксельные координаты
обнаруженные углы (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("RMS : \n")
print(ret)
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

"""
cameraMatrix = [[646.83159356   0.         279.64003368]
                [  0.         643.56638335 254.65526201]
                [  0.           0.           1.        ]]
dist = [[ 1.07170944e-02  1.57335492e+00  5.05514952e-03 -2.96831740e-02 -3.07266710e+01]]


0.8421021087397718
Camera matrix : 

[[656.7382002    0.         302.2583313 ]
 [  0.         646.92684674 247.13654762]
 [  0.           0.           1.        ]]
dist : 

[[ 0.31157188 -1.34319836 -0.0042151  -0.00592606  1.28977951]]

"""