
import numpy as np
import os
from util.read_data import read_pose,read_intrinsic_yaml
import open3d as o3d
from matplotlib import pyplot as plt
import cv2

class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)


def integration(root, CameraIntrinsic):
    files = os.listdir(root)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.001,
            sdf_trunc=0.001,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    count = 0
    for f in files:
        if '.pose' in f:
            if count >=6:
                break
            metadata = f.split('.')[0]
            mat = read_pose(os.path.join(root, f).replace('\\','/'))
            camera_pose = CameraPose(metadata, mat)
            camera_pose.pose[:3,3] =camera_pose.pose[:3,3] * 0.001
            # traj.append(CameraPose(metadata, mat))
            color_file = os.path.join(root, metadata+'.color.jpg').replace('\\','/')
            color = o3d.io.read_image(color_file)
            depth_file = os.path.join(root, metadata+'.depth.png').replace('\\','/')
            depth = o3d.io.read_image(depth_file)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale  = 1000, depth_trunc=0.6, convert_rgb_to_intensity=False)
       
            # plt.imshow(depth)
            # plt.show()
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
   CameraIntrinsic =  read_intrinsic_yaml('camera_params_tsdf.yml')
   root = 'data/tsdf-fusion-test-data/data_uint16'
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
    
