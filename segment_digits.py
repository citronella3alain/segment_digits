#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import sys
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

model_nums = keras.models.load_model('model_nums/')
model_ops = keras.models.load_model('model_ops/')
op_labels = ['/', '*', '+', '-']

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

def perim(frac, dim, draw_on):
    h, w = dim
    # y1, x1, y2, x2
    coords = [h*(1-frac)//2, w*(1-frac)//2, h*(frac+1)//2, w*(frac+1)//2]
    coords = [int(c) for c in coords]
    cv2.rectangle(draw_on, (coords[1], coords[0]), (coords[3], coords[2]), (0, 255, 0), 5)
    return coords
def condition(x, y, w, h):
    return not (w*h<1000 or x < p_coords[1] or x + w > p_coords[3] or y < p_coords[0] or y > p_coords[2])
def pad2square(img, side_len, x, y, w, h):
    x1 = x+w
    y1 = y+h
    char_box = img[y:y1, x:x1]
    if h > w:
        squared_img = np.pad(char_box, [(0,),((h-w)//2,)])
    elif h < w:
        squared_img = np.pad(char_box, [((w-h)//2,), (0, )])
    else:
        squared_img = char_box
    squared_img = np.pad(squared_img, int(.2*max(h, w)))
    resized = cv2.resize(squared_img, (side_len, side_len), interpolation= cv2.INTER_AREA)
    return resized

img_in, cntrs, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

digitCnts = []
final = input_img.copy()  

counter = 0
p_coords = perim(.95, th.shape, final)
bounding_boxes = [cv2.boundingRect(c) for c in cntrs]
bounding_boxes = sorted([box for box in bounding_boxes if condition(*box)], key=lambda b: (b[0], b[1]))

num_inputs = [pad2square(th, 28, *bounding_boxes[0]), pad2square(th, 28, *bounding_boxes[2])]
numbers = np.argmax(model_nums.predict(np.array(num_inputs)), axis=1)

op_input = 255-pad2square(th, 40, *bounding_boxes[1])
op = op_labels[np.argmax(model_ops.predict(np.array([op_input])))]

print(numbers, op)
if op == '+':
    print(sum(numbers))
elif op == '-':
    print(numbers[0] - numbers[1])
elif op == '*':
    print(numbers[0] * numbers[1])
elif op == '/':
    print(numbers[0] // numbers[1])


counter = 0
for box in bounding_boxes:
    x, y, w, h = box
    cv2.rectangle(final,(x,y),(x+w,y+h),(0, 0, 255), 5)
final = cv2.putText(final, f'{numbers[0]}', (bounding_boxes[0][0], bounding_boxes[0][1]+bounding_boxes[0][3]), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 0, 0), 2, cv2.LINE_AA)
final = cv2.putText(final, f'{numbers[1]}', (bounding_boxes[2][0], bounding_boxes[2][1]+bounding_boxes[2][3]), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 0, 0), 2, cv2.LINE_AA)
final = cv2.putText(final, f'{op}', (bounding_boxes[1][0], bounding_boxes[1][1]+bounding_boxes[1][3]), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imwrite(f'ann_{filename}', final)
# for c in cntrs:
#     (x, y, w, h) = cv2.boundingRect(c)
#     if w*h<1000 or x < p_coords[1] or x + w > p_coords[3] or y < p_coords[0] or y > p_coords[2]:
#         continue
#     # print(counter, w*h, x, y, w, h)
#     resized = pad2square(th, 28, x, y, w, h)
#     # norm_image = cv2.normalize(resized, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
#     # norm_image = norm_image.reshape((norm_image.shape[0], norm_image.shape[1], 1))
#     # case = np.asarray([norm_image])
#     # pred = model.predict_classes([case])

#     result = np.argmax(model_nums.predict(np.array([resized])))
#     cv2.imwrite(f'obj_{counter}.png', resized)
#     # np.save(f'obj_{counter}.npy', resized)

#     # Drawing the selected contour on the original image
#     cv2.rectangle(final,(x,y),(x+w,y+h),(0, 0, 255), 5)
#     final = cv2.putText(final, f'{result}', (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 0, 0), 2, cv2.LINE_AA)
#     counter += 1
# cv2.imwrite(f'ann_{filename}', final)
