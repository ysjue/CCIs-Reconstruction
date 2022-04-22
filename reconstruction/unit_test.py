from util import read_pose
import numpy as np
import ros2_aruco.transformations as t

poses = read_pose('/home/sean/laser_ws/data/poses.yaml')
q = t.quaternion_inverse([0.9599221300494553, 0.1879986810568463, 0.03995007622508742 ,0.2039852729287404])
m = t.quaternion_matrix(q)
tra = np.asarray([0.1332795306358483 ,-0.058121474751785196 ,0.15391531689594518 ])
print (np.concatenate([np.matmul(m[:3,:3], -1*tra[:,None])[:,0], q]))

exit()
for i, pose in zip( range(len(poses)), poses):
    print("this is the {0:}-th poses:".format(i))
    print( "  -  base2hand: ", pose ) 
    base2cam = np.asarray([[-0.99833985 , 0.03621568,  0.044788 ,  -0.08511892],
                [-0.01750416 ,-0.93158318 , 0.36310656 ,-0.34748815],
                [ 0.0548739  , 0.36171978 , 0.93067054 ,-0.25192905],
                [ 0.  ,        0.     ,     0.    ,      1.        ]])
    hand2base = t.quaternion_matrix(t.quaternion_inverse([0.1332795306358483 ,-0.058121474751785196 ,0.15391531689594518 ,0.9599221300494553, 0.1879986810568463, 0.03995007622508742 ,0.2039852729287404]))
    hand2base[:3,3] = np.matmul(hand2base[:3,:3],-1*pose[:3][:,None])[:,0]
    print(hand2base)
    # hand2base = np.linalg.inv(base2hand)
   
    hand2cam = np.matmul(hand2base, base2cam)
    quat = t.quaternion_from_matrix(hand2cam)
    ros_hand2cam = np.concatenate([hand2cam[:3,3],quat])
    print( "  - hand2cam: ", ros_hand2cam )

