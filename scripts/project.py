import os
import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt

PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = 'data'
RESULT_PATH = 'result'
# CAMERA_MODEL = image_geometry.PinholeCameraModel()
CAMERA_MATRIX = np.array([[1085.8801, 0, 1255.37351],
                          [0, 1087.46558, 747.00803],
                          [0, 0, 1]], dtype=np.float32)

DIST_COEFFS = np.array([-0.084180, 0.000464, 0.000143, -0.001763, 0.000000], dtype=np.float32)



def project(pc_path, img_path, transform_path):
    pcd = o3d.io.read_point_cloud(pc_path)
    img = cv2.imread(img_path)

    data = np.load(transform_path)
    r = data['r']
    t = data['t']

    # transformation_matrix = np.eye(4)
    # transformation_matrix[:3, 3] = T
    # transformation_matrix[:3, :3] = R

    points3D = np.asarray(pcd.points)
    # pcd.transform(transformation_matrix)



    
    points2D, _ = cv2.projectPoints(points3D, r, t, CAMERA_MATRIX, DIST_COEFFS)



    inrange = np.where((points2D[:,:, 0] >= 0) &
                       (points2D[:,:, 1] >= 0) &
                       (points2D[:,:, 0] < img.shape[1]) &
                       (points2D[:,:, 1] < img.shape[0]))
    points2D = points2D[inrange[0]].round().astype('int')



    # Draw the projected 2D points
    for i in range(len(points2D)):
        cv2.circle(img, tuple(points2D[i,0]), 1, (255, 255, 255), -1)
    
    if img is not None:
        disp = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        # cv2.imshow('Image', img)
        # Setup matplotlib GUI
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Project Points')
        ax.set_axis_off()
        ax.imshow(disp)
        plt.show()
    
    else:
        print('Failed to show the image.')

    


if __name__ == '__main__':
    img_path = os.path.join(PKG_PATH, os.path.join(DATA_PATH, "test.png"))
    pc_path = os.path.join(PKG_PATH, os.path.join(DATA_PATH, "test.pcd"))
    transform_path = os.path.join(PKG_PATH, os.path.join(RESULT_PATH, "extrinsics.npz"))

    project(pc_path, img_path, transform_path)