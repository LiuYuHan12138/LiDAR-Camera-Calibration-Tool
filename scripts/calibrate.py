import os
import cv2
import numpy as np
import multiprocessing
import open3d as o3d
import matplotlib.pyplot as plt
import random
from tf.transformations import euler_from_matrix

PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = 'data'
RESULT_PATH = 'result'

CAMERA_MATRIX = np.array([[1085.8801, 0, 1255.37351],
                          [0, 1087.46558, 747.00803],
                          [0, 0, 1]], dtype=np.float32)

DIST_COEFFS = np.array([-0.084180, 0.000464, 0.000143, -0.001763, 0.000000], dtype=np.float32)

def save_data(data, filename, folder):
    # Empty data
    if not len(data): return

    # Handle filename
    filename = os.path.join(PKG_PATH, os.path.join(folder, filename))
    
    # Create folder
    try:
        os.makedirs(os.path.join(PKG_PATH, folder))
    except OSError:
        if not os.path.isdir(os.path.join(PKG_PATH, folder)): raise

    # Save points data
    if os.path.isfile(filename):
        data = np.vstack((np.load(filename), data))
    np.save(filename, data)
    

def extract_points_2D(path):

    print(PKG_PATH)
    
    img = cv2.imread(path)
    disp = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
	# Setup matplotlib GUI
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Select 2D Image Points')
    ax.set_axis_off()
    ax.imshow(disp)
    corners = []

    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'pink', 'gray', 'cyan']

    def onclick(event):
        x = event.xdata
        y = event.ydata
        if (x is None) or (y is None): return

        # Display the picked point
        corners.append((x, y))
        print("IMG pick:", str(corners[-1]))

        ax.plot(x, y, 'o', color=colors[random.randint(0, 9)], label='Points')
        ax.figure.canvas.draw_idle()

	# Display GUI
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    print(corners)

    if (len(corners) >= 6):
        save_data(corners, 'img_corners.npy', RESULT_PATH)
    else:
        print('PnP Requires minimum 6 points')
        return
    
    # save_data(corners, 'img_corners.npy', RESULT_PATH)
    


def extract_points_3D(path):
    pcd = o3d.io.read_point_cloud(path)
    intensity = np.asarray(pcd.colors)[:, 0]


    # 设置点云颜色
    pcd.paint_uniform_color([1.0, 1.0, 1.0]) 
    
    if len(np.asarray(pcd.points)) > 5:
        print('PCL points available: %d', len(np.asarray(pcd.points)))
    else:
        print('Very few PCL points available in range')
        return
	
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(width=800, height=600, left=500, top = 200)
    vis.add_geometry(pcd)

    # 获取渲染选项并设置背景色为黑色
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0, 0, 0])  # 设置为黑色
    render_option.point_size = 1.5


    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    selected_points = vis.get_picked_points()
    

    points = np.asarray(pcd.points)[selected_points]
    print(points)
    # print("3D points:")
    # print(points)
    if (points.shape[0] >= 6):
        save_data(points, 'pcl_corners.npy', RESULT_PATH)
    else:
        print('PnP Requires minimum 6 points')
        return


def calibrate(points2D = None, points3D = None):
    folder = os.path.join(PKG_PATH, RESULT_PATH)
    if points2D is None: points2D = np.load(os.path.join(folder, 'img_corners.npy'))
    if points3D is None: points3D = np.load(os.path.join(folder, 'pcl_corners.npy'))

	# Check points shape
    assert(points2D.shape[0] == points3D.shape[0])
    if not (points2D.shape[0] >= 6):
        print('PnP Requires minimum 6 points')
        return
    
    
    
    success, rotation_vector, translation_vector = cv2.solvePnP(points3D, points2D, 
        CAMERA_MATRIX, DIST_COEFFS)
    
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    euler = euler_from_matrix(rotation_matrix)
    
    if(success):
        print('Rotation vector:', rotation_vector.T)
        print('Rotation Matrix:', rotation_matrix)
        print('euler:', euler)
        print('Translation Vector:', translation_vector.T)

        with open(os.path.join(folder, 'extrinsic.txt'), 'a') as file:
            np.savetxt(file, rotation_vector.T, header='Rotation vector:', fmt='%f', comments='')
            file.write('\n')
            np.savetxt(file, rotation_matrix, header='Rotation Matrix:', fmt='%f',  comments='')
            file.write('\n')
            np.savetxt(file, euler, header='euler:', fmt='%f', comments='')
            file.write('\n')
            np.savetxt(file, translation_vector.T, header='Translation Vector:', fmt='%f', comments='')
        
        np.savez(os.path.join(folder, 'extrinsics.npz'), r = rotation_vector.T, R=rotation_matrix, euler=euler, t=translation_vector.T, allow_pickle=True)
    
    else:
        print("Failed to solve pnp")


if __name__ == '__main__':
    
    image_path = os.path.join(PKG_PATH, os.path.join(DATA_PATH, 'test.png'))
    pointcloud_path = os.path.join(PKG_PATH, os.path.join(DATA_PATH, 'test.pcd'))
    
    # pick points

    img_p = multiprocessing.Process(target=extract_points_2D, args= [image_path])
    pcl_p = multiprocessing.Process(target=extract_points_3D, args= [pointcloud_path])
    img_p.start(); 
    pcl_p.start()
    img_p.join(); 
    pcl_p.join()
    
	# calibrate

    calibrate()

