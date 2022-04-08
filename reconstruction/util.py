import yaml
import numpy as np
from ros2_aruco.transformations import quaternion_matrix 

def read_pose(dir):
    """The 'robot' key saves the W frame w.r.t. T frame, and we claculate the Transform_W_S by 
        using Tranform_W_T * Transformation_T_S

    """
    with open(dir, 'r') as file:
        samples = yaml.safe_load(file)['samples']
    poses = np.ones((len(samples),7), dtype=np.float32)
    for i, sample in zip(range(len(samples)), samples):
        translation = [sample['robot']['translation']['x'],
                        sample['robot']['translation']['y'],
                        sample['robot']['translation']['z']]
        quat = [sample['robot']['rotation']['x'],
                sample['robot']['rotation']['y'],
                sample['robot']['rotation']['z'],
                sample['robot']['rotation']['w']]
        poses[i] = np.asarray(translation + quat, dtype=np.float32)
    return poses
        
    

if __name__ ==  '__main__':
    read_pose('/home/sean/laser_ws/data/poses.yaml')