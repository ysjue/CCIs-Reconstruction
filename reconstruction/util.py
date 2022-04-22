import yaml
import numpy as np
import cv2
from scipy import ndimage
from ros2_aruco.transformations import quaternion_matrix 

def read_pose(dir, field = 'robot'):
    """The 'robot' field saves the hand frame w.r.t. base frame (base2hand), 
    and following claculation of transformation hand2cam is needed.
    """
    with open(dir, 'r') as file:
        samples = yaml.safe_load(file)['samples']
    poses = np.ones((len(samples),7), dtype=np.float32)
    for i, sample in zip(range(len(samples)), samples):
        translation = [sample[field]['translation']['x'],
                        sample[field]['translation']['y'],
                        sample[field]['translation']['z']]
        quat = [sample[field]['rotation']['x'],
                sample[field]['rotation']['y'],
                sample[field]['rotation']['z'],
                sample[field]['rotation']['w']]
        poses[i] = np.asarray(translation + quat, dtype=np.float32)
    return poses

def read_intrinsic(dir):
    CameraIntrinsic = {}
    with open(dir, 'r') as file:
        param = yaml.safe_load(file)
        CameraIntrinsic['image_height'] = param['height']
        CameraIntrinsic['image_width'] = param['width']
        k = param['k']
    CameraIntrinsic['fx'] = k[0]
    CameraIntrinsic['fy'] = k[4]
    CameraIntrinsic['cx'] = k[2]
    CameraIntrinsic['cy'] = k[5]
    return CameraIntrinsic

def get_largest_one_component(img, print_info = False, threshold = None):
    """
    Get the largest two components of a binary volume
    inputs:
        img: the input 2D image
        threshold: a size threshold
    outputs:
        out_img: the output image 
    """
    s = ndimage.generate_binary_structure(2,2) # iterate structure
    labeled_array, numpatches = ndimage.label(img,s) # labeling
    sizes = ndimage.sum(img, labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if(print_info):
        print('component size', sizes_list)
    if(len(sizes) == 1):
        out_img = img
    else:
        if(threshold):
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:    
            max_size1 = sizes_list[-1]
            #max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            #max_label2 = np.where(sizes == max_size2)[0] + 1
            component1 = labeled_array == max_label1
            #component2 = labeled_array == max_label2
            # if(max_size2*10 > max_size1):
            #     component1 = (component1 + component2) > 0
            out_img = component1
    return np.asarray(out_img)

def fill_holes(img):
    """
    filling small holes of a binary volume with morphological operations
    """
    neg = 1 - img
    s = ndimage.generate_binary_structure(2,1) # iterate structure
    labeled_array, numpatches = ndimage.label(neg,s) # labeling
    sizes = ndimage.sum(neg,labeled_array,range(1,numpatches+1)) 
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    max_size = sizes_list[-1]
    max_label = np.where(sizes == max_size)[0] + 1
    component = labeled_array == max_label
    return 1 - component




if __name__ ==  '__main__':
    read_pose('/home/sean/laser_ws/data/poses.yaml')
    print(read_intrinsic('/home/sean/laser_ws/data/intrinsic.yaml'))
