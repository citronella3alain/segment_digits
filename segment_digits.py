#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

filename = sys.argv[1]
# cv2.namedWindow('output', cv2.WINDOW_NORMAL)
input_img = cv2.imread(filename)
# cv2.imshow('output', input_img)
img_blurred = cv2.GaussianBlur(cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY), (5,5), 0)
# cv2.imshow('output', img_blurred)
# cv2.waitKey(0) 

ret, th = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cv2.imshow('output', th)
# cv2.waitKey(0) 

# output = cv2.connectedComponentsWithStats(th, 8, cv2.CV_32S)
# (numLabels, labels, stats, centroids) = output

# cv2.imshow('output', labels)
# cv2.waitKey(0)
# plt.imshow(labels)
# print(labels)

img_in, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

digitCnts = []
areas = []
final = input_img.copy()  

# final_h, final_w = final.shape
def perim(frac, dim, draw_on):
    h, w = dim
    # y1, x1, y2, x2
    coords = [h*(1-frac)//2, w*(1-frac)//2, h*(frac+1)//2, w*(frac+1)//2]
    coords = [int(c) for c in coords]
    cv2.rectangle(draw_on, (coords[1], coords[0]), (coords[3], coords[2]), (0, 255, 0), 5)
    return coords

counter = 0
# print(hierarchy)
p_coords = perim(.95, th.shape, final)
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    if w*h<1000 or x < p_coords[1] or x + w > p_coords[3] or y < p_coords[0] or y > p_coords[2]:
        continue
    # print(counter, w*h, x, y, w, h)
    areas.append(w*h)
    # if the contour is sufficiently large, it must be a digit
#     if (w >= 20 and w <= 290) and h >= (th3.shape[0]>>1)-15:
    x1 = x+w
    y1 = y+h
    xmid = x + w//2
    ymid = y + w//2
    digitCnts.append([x,x1,y,y1])
    # Drawing the selected contour on the original image
    cv2.rectangle(final,(x,y),(x1,y1),(0, 0, 255), 5)
    final = cv2.putText(final, f'{counter}', (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 0, 0), 2, cv2.LINE_AA)
    counter += 1
cv2.imwrite(f'ann_{filename}', final)
