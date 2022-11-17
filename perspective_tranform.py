import cv2
import numpy as np
import matplotlib.pyplot as plt
import lib
import os
import argparse
from pathlib import Path

def main(args):
    img_path = Path(args.input)
    save_path = Path(args.output)
    if save_path.stem == '':
        new_name = str(img_path.stem) + '_output' + str(img_path.suffix)
        save_path = img_path.parent / new_name 
    print(img_path)
    img = cv2.imread(str(img_path))
    h,w,d = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ret, bw = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

    bw_erode = cv2.erode(bw,kernel,iterations = 2)

    mask = np.zeros((h, w), np.uint8)
    im_floodfill = bw_erode.copy()
    cv2.floodFill(im_floodfill, None, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = bw_erode | im_floodfill_inv


    im_out_biggest = lib.find_biggest(im_out)

    dst = cv2.cornerHarris(im_out_biggest,50,15,0.005)
    corner_obj = np.zeros(bw_erode.shape[:2])
    corner_obj[dst>0.005*dst.max()]=[255]
    corner_obj = corner_obj.astype(np.uint8)
    corner = lib.find_object(corner_obj) #y,x (row,col)
    corner_4 = lib.max_area(corner)

    corner_pic = corner_obj.copy()
    corner_pic  = cv2.circle(corner_pic, (corner_4[0][0],corner_4[0][1]), 10, 0, -1)
    corner_pic  = cv2.circle(corner_pic, (corner_4[1][0],corner_4[1][1]), 10, 0, -1)
    corner_pic  = cv2.circle(corner_pic, (corner_4[2][0],corner_4[2][1]), 10, 0, -1)
    corner_pic  = cv2.circle(corner_pic, (corner_4[3][0],corner_4[3][1]), 10, 0, -1)

    pts1 = np.float32(lib.assign_corner(corner_4,img))
    pts2 = np.float32([[w,0],[0,0],[0,h],[w,h]])

    matrix = lib.transform(pts1,pts2)
    output = cv2.warpPerspective(img, matrix, (w,h))
    cv2.imwrite(str(save_path) ,output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='path to input image')
    parser.add_argument('--output', help='path to save output image', default='')
    args = parser.parse_args()
    main(args)