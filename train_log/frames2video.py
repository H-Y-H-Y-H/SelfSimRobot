import cv2
import numpy as np
import glob

img_array = []
for i in range(0,8000,25):
    filename = 'log_debug/image/%d.png'%i
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()