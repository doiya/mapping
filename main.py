# coding:utf-8

import numpy as np
import cv2


### 画像分割
def img_split(src_img, grid_w, grid_h):
    h, w = src_img.shape[:2]
    dst_img = []
    inter_w, inter_h = int(w/(grid_w*2-1)), int(h/(grid_h*2-1))
    for y in range(grid_h*2-1):
        retu_img = []
        for x in range(grid_w*2-1):
            retu_img.append(src_img[y*inter_h:(y+1)*inter_h, x*inter_w:(x+1)*inter_w])
        dst_img.append(retu_img)
    return dst_img


### マーカーから座標取得
def detectMarkers_pos(img, grid_w, grid_h, aruco, dictionary):
    ### 座標, ID
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)

    ### 配列を初期化
    sPoint = np.zeros((grid_h*2, grid_w*2, 2), np.int32)

    for i, corner in enumerate( corners ):
        points = corner[0].astype(np.int32)
        #cv2.polylines(img, [points], True, (0,255,0), 2)
        #cv2.putText(img, str(ids[i][0]), tuple(points[0]), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255), 2)

        n = ids[i][0]
        ### マーカーの4点を使用
        sPoint[int(n/grid_w) * 2][n%grid_w * 2] = points[0]
        sPoint[int(n/grid_w) * 2][n%grid_w * 2 + 1] = points[1] + [1, 0]
        sPoint[int(n/grid_w) * 2 + 1][n%grid_w * 2] = points[3] + [0, 1]
        sPoint[int(n/grid_w) * 2 + 1][n%grid_w * 2 + 1] = points[2] + [1, 1]

    return sPoint


### ホモグラフィー
### img:マッピングされる画像  src_img:分割されたマッピングする画像  rect:変換後の座標
def homography(img, src_img, rect):
    height, width  = src_img.shape[:2]
    sPoints = np.array([
        [width, 0],
        [0,0],
        [0, height],
        [width, height]])
    pts1 = np.float32(sPoints)
    pts2 = np.float32(rect)
    M = cv2.getPerspectiveTransform(pts1,pts2)

    # 画像サイズの取得(横, 縦)
    size = tuple([img.shape[1], img.shape[0]])

    img = cv2.warpPerspective(src_img, M, size, img, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    return img


### オーバーレイ
### img:マッピングされる画像  overlay_img:マッピングする画像
def overlayOnPart(img, split_img, grid_w, grid_h, aruco, dictionary):
    ### マーカーから座標取得
    marker_pos = detectMarkers_pos(img, grid_w, grid_h, aruco, dictionary)
    ### 射影変換
    for y in range(grid_h*2-1):
        for x in range(grid_w*2-1):
            # 座標取得
            rect = np.array([
                marker_pos[y][x+1],
                marker_pos[y][x],
                marker_pos[y+1][x],
                marker_pos[y+1][x+1]
            ])

            ### rectに(0,0)があればpass
            if [0,0] in rect:
                pass
            else:
                rect = rect + np.array([[1, -1], [-1, -1], [-1, 2], [1, 2]])
                img = homography(img, split_img[y][x], rect)
    return img



def main():
    ### オーバーレイ用の画像
    #overlay_img = cv2.imread('neko.png', -1)
    overlay_img = cv2.imread('bubka.jpg')

    ### 画像分割
    grid_w, grid_h = 6, 8
    split_img = img_split(overlay_img, grid_w, grid_h)

    ### ArUco
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    '''# 画像テスト
    filename = 'marker.png'
    #filename = 'test.png'
    img = cv2.imread(filename)
    img = overlayOnPart(img, split_img, grid_w, grid_h, aruco, dictionary)
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    # カメラのID
    DEVICE_ID = 0
    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)
    #cap = cv2.VideoCapture('test.mp4')

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == False:
            break
        else:
            frame = overlayOnPart(frame, split_img, grid_w, grid_h, aruco, dictionary)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
