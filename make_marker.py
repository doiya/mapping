# coding:utf-8

import cv2
import numpy as np


aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

grid_w = 6
grid_h = 8

width = 1200
height = 1600

dst_img = np.tile(np.uint8([255]), (height, width))

# マーカーを生成
marker = []
for i in range(grid_w * grid_h):
    marker.append(aruco.drawMarker(dictionary, i, 100))

# マーカーボードを生成
for y in range(grid_h):
    for x in range(grid_w):
        dst_img[y*200 + 50:y*200 + 150:, x*200 + 50:x*200 + 150:] = marker[x + grid_w * y]

cv2.imshow('dst', dst_img)
cv2.imwrite('marker.png', dst_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
