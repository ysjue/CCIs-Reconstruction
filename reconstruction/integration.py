
from cv2 import CV_16UC1, CV_8UC1, CV_8UC3
from matplotlib.transforms import Transform
import numpy as np
import os
from util import read_pose, read_intrinsic
import open3d as o3d
from matplotlib import pyplot as plt
import cv2
from ros2_aruco.transformations import quaternion_matrix, quaternion_inverse

global using_mask
using_mask = True
class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)


def integration(root, CameraIntrinsic=None):
    files = os.listdir(root)
    base2cam = np.asarray([[-0.99833985 , 0.03621568,  0.044788 ,  -0.08876],
                [-0.01750416 ,-0.93158318 , 0.36310656 ,-0.317585],
                [ 0.0548739  , 0.36171978 , 0.93067054 ,-0.251979],
                [ 0.  ,        0.     ,     0.    ,      1.        ]])
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.0008,
            sdf_trunc=0.003,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    count = 0
    ros_poses = read_pose(os.path.join(root, 'poses.yaml').replace('\\','/'), 'tracking')
    for i in range(11):
        # if i !=7:
        #     continue
        # if count != 4 and count != 7:
        #     count += 1 
        #     continue

        metadata = str(i)
        # metadata = f.split('_')[0]
        pose = ros_poses[int(metadata)]
        # Transform_base_W ----> Transform_W_base
        base2hand =  quaternion_matrix(pose[3:7])    # rotation
        base2hand[:3,3] = pose[:3]    # translation (m)
        hand2base = np.linalg.inv(base2hand)
        # calculate camera frame w.r.t. end effector frame (W), i.e., Transform_hand2cam
        mat = np.matmul(hand2base, base2cam)
        # mat =  hand2cam
        camera_pose = CameraPose(metadata, mat)
        camera_pose.pose[:3,3] =camera_pose.pose[:3,3] # * 0.001
        if using_mask:
            mask_file = os.path.join(root, metadata+'_annotation.jpg').replace('\\','/')
            mask = cv2.imread(mask_file,CV_8UC1)
            color_file = os.path.join(root, metadata+'_color.jpg').replace('\\','/')
            
            color_img = cv2.imread(color_file)
            color_img[mask == 2 ] = [0,0,0]
            
            color = o3d.geometry.Image(color_img.astype(np.uint8))
            
            depth_file = os.path.join(root, metadata+'_depth.png').replace('\\','/')
            depth_img = cv2.imread(depth_file, CV_16UC1)
            depth_img[mask == 2 ] = 0
            

            plt.imshow(depth_img)
            plt.show()
            depth = o3d.geometry.Image(depth_img.astype(np.uint16))
        else:
            color_file = os.path.join(root, metadata+'_color.jpg').replace('\\','/')
            depth_file = os.path.join(root, metadata+'_depth.png').replace('\\','/')
            color = o3d.io.read_image(color_file)
            depth = o3d.io.read_image(depth_file)
        
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale  = 10000, depth_trunc=0.3, convert_rgb_to_intensity=False)
        

        volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(CameraIntrinsic['image_width'], CameraIntrinsic['image_height'],\
                                    CameraIntrinsic['fx'], CameraIntrinsic['fy'],\
                                    CameraIntrinsic['cx'], CameraIntrinsic['cy']),
                # o3d.camera.PinholeCameraIntrinsic(
                #     o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
                np.linalg.inv(camera_pose.pose))
        
        print('{0:}: '.format(count))
        print(camera_pose)
        count += 1 
        # if count > 12:
        #     break

    return volume


if __name__ == '__main__':
   CameraIntrinsic =  read_intrinsic('/home/sean/laser_ws/intrinsic.yaml')
   root = '/home/sean/laser_ws/data1'
   volume = integration(root, CameraIntrinsic)
#    pt_cloud = volume.extract_point_cloud()
   print("Extract a triangle mesh from the volume and visualize it.")
   mesh = volume.extract_triangle_mesh()
   mesh.compute_vertex_normals()
   o3d.visualization.draw_geometries([mesh])
#                                     front=[0.5297, -0.1873, -0.8272],
#                                     lookat=[2.0712, 2.0312, 1.7251],
#                                     up=[-0.0558, -0.9809, 0.1864],
#                                   zoom=0.47)
    
