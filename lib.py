import numpy as np
import cv2
from itertools import permutations

def find_object(img):
    img_copy = img.copy()
    img_ = img.copy()
    white_obj = {}
    i_object = 0
    object = []
    while sum(sum(img_copy)) != 0:
        white_y,white_x = np.where(img_copy == 255)
        cv2.floodFill(img_, None, (white_x[0], white_y[0]), 0)
        temp = img_copy - img_
        white_obj[i_object] = temp
        i_object = i_object + 1
        cv2.floodFill(img_copy,None,(white_x[0],white_y[0]), 0)
    for i in range(len(white_obj)):
        white_y, white_x = np.where(white_obj[i] == 255)
        white_y = int(np.mean(white_y))
        white_x = int(np.mean(white_x))
        object.append((white_x,white_y))
    return np.array(object)

def verify_corner(corner,img):
    h,w = img.shape[:2]
    pts = np.float32([[0,0],[0,w],[h,w],[h,0]])
    pts2 = np.array([])
    pts2 = []
    for i in pts:
        dummy = []
        for j in corner:
            print(i,j)
            distance = np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2)
            # print(distance)
            # dummy = np.append(dummy,distance)
            dummy.append([distance])
        pts2.append(corner[np.argmin(dummy)])
    return np.float32(pts2)

def transform(src,dst):
    a = np.zeros([8,8])
    b = np.zeros([8,1])
    for i in range(0,4):
        a[i][0] = a[i + 4][3] = src[i][0];
        a[i][1] = a[i + 4][4] = src[i][1];
        a[i][2] = a[i + 4][5] = 1;
        # a[i][3] = a[i][4] = a[i][5] = a[i + 4][0] = a[i + 4][1] = a[i + 4][2] = 0;
        a[i][6] = -src[i][0] * dst[i][0];
        a[i][7] = -src[i][1] * dst[i][0];
        a[i + 4][6] = -src[i][0] * dst[i][1];
        a[i + 4][7] = -src[i][1] * dst[i][1];
        b[i] = dst[i][0];
        b[i + 4] = dst[i][1];
    x = np.linalg.solve(a, b)
    x = np.append(x,1)
    x = x.reshape(3,3)
    return x

def area(coordinate):
    coordinate = np.append(coordinate,coordinate[[0]],axis=0)
    area = 0.5*abs((sum(coordinate[:-1,0] * coordinate[1:,1]) - sum(coordinate[:-1,1] * coordinate[1:,0])))
    return area

def max_area(corner):
    perm = np.array(list(permutations(corner, 4)))
    max = 0
    for i in perm:
        area_of_rect = area(i)
        if area_of_rect > max:
            output = i
            max = area_of_rect
    return output

def find_biggest(img):
    img_copy = img.copy()
    img_ = img.copy()
    white_obj = {}
    i_object = 0
    corner = []
    max = 0
    while sum(sum(img_copy)) != 0:
        white_y, white_x = np.where(img_copy == 255)
        cv2.floodFill(img_, None, (white_x[0], white_y[0]), 0)
        temp = img_copy - img_
        white_obj[i_object] = temp
        i_object = i_object + 1
        cv2.floodFill(img_copy, None, (white_x[0], white_y[0]), 0)
    for i in range(len(white_obj)):
        white_y, white_x = np.where(white_obj[i] == 255)
        if np.size(white_y) > max:
            max = np.size(white_y)
            biggest = white_obj[i]
    return biggest

def assign_corner(corner,img):
    h, w = img.shape[:2]
    center = [int(w/2),int(h/2)] ## [xc,yc]
    deg = []
    for l in corner:## [xl,yl]
        c = np.arctan2(center[1]-l[1],l[0]-center[0])
        if c>0:
            deg.append(c*(180/np.pi))
        else:
            deg.append(c * (180 / np.pi)+360)
    deg = np.array([deg])
    deg = np.transpose(deg)
    corner = np.concatenate((corner, deg), axis=1)
    corner = corner[corner[:,2].argsort()]
    corner = np.delete(corner,2,1)
    condition1 = np.sqrt((corner[1][0]-corner[0][0])**2 + (corner[1][1]-corner[0][1])**2)
    condition2 = np.sqrt((corner[3][0]-corner[0][0])**2 + (corner[3][1]-corner[0][1])**2)
    if condition2>condition1:
        corner = corner[[1,2,3,0],:]
    return corner













































