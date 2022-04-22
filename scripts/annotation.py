#!/usr/bin/env python
from cv2 import EVENT_MOUSEMOVE
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from reconstruction.util import get_largest_one_component


draw_mode = False # true if mouse is pressed
global class_id
class_id = 1
# mouse callback function

def draw_circle(event,x,y,flags,param):
    kernal_size = 3
    global draw_mode, img, mask, np_img
 
    if event== cv.EVENT_MOUSEMOVE and draw_mode:
        for i in range(2 * kernal_size):
            for j in range(2 * kernal_size):
                x_i = max(min(x - kernal_size + j, shape[1] -1 ), 0)
                y_i = max(min(y - kernal_size + i, shape[0] -1 ), 0)
                mask[y_i,x_i] = class_id
                scalar = [0]*3
                scalar[-class_id] = 255 
                img[y_i,x_i] = scalar
    elif event == cv.EVENT_LBUTTONDOWN:
        draw_mode = not draw_mode
        


root = '/home/sean/laser_ws/transparent_data/'
files = os.listdir(root)
for file in files:
    if 'color.jpg' not in file:
        continue
    file_dir = os.path.join(root, file)
    img = cv.imread(file_dir)
    origin_img = img.copy()
    shape = img.shape[:-1]
    mask = np.zeros(img.shape[:-1], np.uint8)
    cv.namedWindow(file)
   
    cv.setMouseCallback(file,draw_circle)
    while(1):
        
        cv.imshow(file,img)
        
        k = cv.waitKey(1) & 0xFF
        # if k!=255:
        #     print(k)
        if k < 52 and k> 48:
            class_id = k - 48
            print('Annotate the label of class {0:}'.format(class_id))
    
        elif k == 27: 
            cv.destroyAllWindows()
            draw_mode = False
            break
        
        elif k == 100: # Press key d
            img = origin_img.copy()
            mask = np.zeros(img.shape[:-1], np.uint8)
            draw_mode = False
            print('delete current annotaion')
        elif k == 119: # Press 'w', denoting write
            
            write_path = os.path.join(root, file.replace('_color.jpg', '_annotation.jpg'))
            cv.imwrite(write_path, mask)
            print(file + ' has been saved to '+write_path)
            draw_mode = False
            cv.destroyAllWindows()
            break
        
        elif k == 115:  # Press 's', denoting segmentation
            draw_mode =False
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            
            pred = np.zeros(img.shape,np.uint8)
            seg = cv.GC_PR_BGD * np.ones_like(mask)
            seg[mask == 2] = 0
            seg[mask == 1] = 1
            
            
            cv.grabCut(origin_img, seg, None, bgdModel,fgdModel,9)
            
            seg_mask = np.where((seg==2)|(seg==0),0,1).astype('uint8')
            seg_mask = get_largest_one_component(seg_mask)
            
            plt.imshow(seg_mask)
            plt.show()
            mask[seg_mask] = 1
            mask[seg_mask == False] = 2
            alpha = 0.6
            beta = (1.0 - alpha)
            img = origin_img.copy()
            pred[seg_mask>0] = [0,0,255]
 
            img = cv.addWeighted(img, alpha, pred, beta, 0.0)
            

            # cv.destroyAllWindows()
            # break
    
cv.destroyAllWindows()
    





