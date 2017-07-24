# coding:utf-8

import numpy as np
import cv2
#from homography import homography


# 画像分割
def img_split(src_img, grid_w, grid_h):
    w, h = src_img.shape[:2]
    dst_img = []
    inter_w, inter_h = int(w/grid_w), int(h/grid_h)
    for y in range(grid_h):
        retu_img = []
        for x in range(grid_w):
            retu_img.append(src_img[y*inter_h:(y+1)*inter_h, x*inter_w:(x+1)*inter_w])
        dst_img.append(retu_img)
    return dst_img


# マーカーから座標取得
def detectMarkers_pos(img, grid_w, grid_h):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    # 座標, ID
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)
    #print (corners)

    # 配列を初期化
    sPoints = [ [[0]*2] * (grid_w+1) ] * (grid_h+1)
    #sPoints = [ [[0]*2] * (grid_w+1) ] * (grid_h+1)  '''マーカーの4点を使用'''
    np.array(sPoints)
    #print (len(sPoints[0]))

    tmp = [[0]*2] * ((grid_w+1)*(grid_h+1))

    for i, corner in enumerate( corners ):
        points = corner[0].astype(np.int32)
        #print (points)
        #cv2.polylines(img, [points], True, (0,255,0), 2)
        #cv2.putText(img, str(ids[i][0]), tuple(points[0]), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255), 2)

        n = ids[i][0]
        #tmp[n] = points[0]  # point[0] マーカーの左上の座標
        '''マーカーの4点を使用'''
        w = int((grid_w + 1) / 2)
        h = int((grid_h + 1) / 2)
        tmp[int(n/w) * w * 4 + (n%w * 2)] = points[0]
        tmp[int(n/w) * w * 4 + (n%w * 2) + 1] = points[1] + [1, 0]
        tmp[int(n/w) * w * 4 + (n%w * 2) + w*2] = points[3] + [0, 1]
        tmp[int(n/w) * w * 4 + (n%w * 2) + w*2 + 1] = points[2] + [1, 1]
        #print (n)
        #print (tmp[int(n/w) * w * 4 + (n%w * 2)])
        #print (tmp[(int(n/w) * w * 4 + (n%w * 2) + 1)])
        #print (tmp[(int(n/w) * w * 4 + (n%w * 2) + w*2)])
        #print (tmp[(int(n/w) * w * 4 + (n%w * 2) + w*2 + 1)])

        #sPoints[int(n/(grid_w+1))][n%(grid_w+1)] = points[0]
        #print (str(int(n/(grid_w+1)))+str(n%(grid_w+1)))
        #print (points[0])
    #print (tmp)
    for y in range(grid_h+1):
        sPoints[y] = tmp[y*(grid_w+1):(y+1)*(grid_w+1)]
        #print (sPoints[y])

    #print (sPoints)
    return sPoints


# 射影変換
def homography(cap, src_img, rect):
    height, width  = src_img.shape[:2]
    sPoints = np.array([
        [width, 0],
        [0,0],
        [0, height],
        [width, height]])
    #print (rect)
    pts1 = np.float32(sPoints)
    pts2 = np.float32(rect)
    #print(pts1)
    #print(pts2)
    M = cv2.getPerspectiveTransform(pts1,pts2)

    # 画像サイズの取得(横, 縦)
    size = tuple([cap.shape[1], cap.shape[0]])

    # dst 画像用意
    #dst_img = np.zeros((size[1], size[0], 4), np.uint8)

    cap = cv2.warpPerspective(src_img, M, size, cap, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    #img = cv2.warpPerspective(img, M,(cap.shape[1], cap.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    #cv2.imshow('getPerspectiveTransform', cap)
    return cap




# オーバーレイ
def overlayOnPart(frame, overlay_img, grid_w, grid_h):
    # マーカーから座標取得
    marker_pos = detectMarkers_pos(frame, grid_w, grid_h)
    #print (marker_pos)
    # 射影変換
    for y in range(grid_h):
        for x in range(grid_w):
            # 座標取得
            rect = np.array([
                marker_pos[y][x+1],
                marker_pos[y][x],
                marker_pos[y+1][x],
                marker_pos[y+1][x+1]
                #marker_pos[y][x+1] + [1, -1],
                #marker_pos[y][x] + [-1, -1],
                #marker_pos[y+1][x] + [-1, 2],
                #marker_pos[y+1][x+1] + [1, 2]
            ])
            #print (str(x)+str(y))
            #print (rect[0])
            #print (rect[1])
            #print (rect[2])
            #print (rect[3])

            # rectに(0,0)があればpass
            if [0,0] in rect:
                pass
            #elif [1, -1] in rect:
            #    pass
            #elif [-1, -1] in rect:
            #    pass
            #elif [-1, 2] in rect:
            #    pass
            #elif [1, 2] in rect:
            #    pass
            else:
                rect = rect + np.array([[1, -1], [-1, -1], [-1, 2], [1, 2]])
                '''print (str(x)+str(y))
                print (rect[0])
                print (rect[1])
                print (rect[2])
                print (rect[3])'''
                frame = homography(frame, split_img[y][x], rect)
    return frame



if __name__ == '__main__':

    # オーバーレイ用の画像
    overlay_img = cv2.imread('neko.png', -1)
    # 画像分割
    #grid_w, grid_h = 5, 7
    '''マーカーの4点を使用'''
    grid_w, grid_h = 11, 15
    split_img = img_split(overlay_img, grid_w, grid_h)
    #for y in range(grid_h):
    #    for x in range(grid_w):
    #        cv2.imshow(str(x)+str(y), dst_img[y][x])


    # 画像テスト
    '''filename = 'marker.png'
    #filename = 'test.png'
    img = cv2.imread(filename)
    img = overlayOnPart(img, overlay_img, grid_w, grid_h)
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    '''# カメラのID
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
            frame = overlayOnPart(frame, overlay_img, grid_w, grid_h)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
