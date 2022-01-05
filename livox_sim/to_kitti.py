import os
import random
import numpy as np
import pandas as pd
import open3d as o3d
from os.path import join as pjoin
from tqdm import tqdm
from PIL import Image
from shutil import copyfile
import threading
from calibration_kitti import Calibration

random.seed(777)
'''
type, truncation, occlusion, alpha, box2d0, box2d1, box2d2, box2d3
height, width, length, x, y, z, rotation_y
image shape 1241x376
training
    ├── calib / keep the same number as ./velodyne
    ├── image_2 // it doesn't matter what kind of images you put, just keep the same numbers as ./velodyne
    ├── label_2 // keep the same number as ./velodyne
    └── velodyne // **put your data**
'''
main_path = '/work/u0610862/livox_sim/simu_data/'
calib_sample = '/home/u0610862/OpenPCDet/livox_sim/calib.txt'
points_path = pjoin(main_path, 'points')
label_path = pjoin(main_path, 'anno')
image_sample = np.zeros((376, 1241, 3), dtype=np.uint8)

# label_map = {'car': 'Car', 'truck': 'Car', 'bus': 'Car',
#              'bimo': 'Cyclist', 'pedestrian': 'Pedestrian'}
label_map = {'car': 'Car', 'bimo': 'Cyclist', 'pedestrian': 'Pedestrian'}

range_limit = [0, -40, -3, 70, 40, 1]

calib = Calibration(calib_sample)

def read_txt(f):
    pnp = pd.read_csv(f, sep=",", header=None).to_numpy().astype(
        np.float32)
    pnp = pnp[np.where(pnp[:, -1] == 1)][:, :4]
    pnp[:, 3] = 0

    pnp = pnp[np.where(pnp[:, 0] < range_limit[3])]
    pnp = pnp[np.where(pnp[:, 1] > range_limit[1])]
    pnp = pnp[np.where(pnp[:, 1] < range_limit[4])]
    pnp = pnp[np.where(pnp[:, 2] > range_limit[2])]
    pnp = pnp[np.where(pnp[:, 2] < range_limit[5])]

    return pnp


def in_range(max, min, v):
    return v < max and v > min

def read_label(f):
    la = pd.read_csv(f, sep=",", header=None)
    det_lines = []
    for l in la.iterrows():
        if l[1][1] not in label_map.keys():
            continue
        ctype = label_map[l[1][1]]
        pos = l[1][2:5].to_numpy().astype(np.float32)

        if not in_range(range_limit[3], range_limit[0], pos[0]) \
                or not in_range(range_limit[4], range_limit[1], pos[1]) \
                or not in_range(range_limit[5], range_limit[2], pos[2]):
            continue

        pos = calib.lidar_to_rect(np.expand_dims(pos, axis=0))[0]
        length = float(l[1][6])
        width = float(l[1][5])
        height = float(l[1][7])
        pos[2] -= height/2.0
        ry = float(l[1][8])

        det_lines.append('%s %d %d %d %d %d %d %d %f %f %f %f %f %f %f\n' % (
            ctype, 0, 0, 0, 1, 2, 51, 52, height, width, length, pos[0], pos[1], pos[2], ry))

    return det_lines


def write_lines(det_lines, f):
    with open(f, 'w') as h:
        h.writelines(det_lines)


if __name__ == '__main__':

    save_dir = pjoin(points_path, '..', 'kitti_sim')
    velodyne_dir = pjoin(save_dir, 'training/velodyne')
    image_2_dir = pjoin(save_dir, 'training/image_2')
    label_2_dir = pjoin(save_dir, 'training/label_2')
    calib_dir = pjoin(save_dir, 'training/calib')
    sets_dir = pjoin(save_dir, 'ImageSets')

    os.makedirs(velodyne_dir, exist_ok=True)
    os.makedirs(image_2_dir, exist_ok=True)
    os.makedirs(label_2_dir, exist_ok=True)
    os.makedirs(sets_dir, exist_ok=True)
    os.makedirs(calib_dir, exist_ok=True)

    pfiles = os.listdir(points_path)
    pfiles.sort()

    frame_list = []

    def make_data(f):
        print(f)
        frame = int(f.split('.')[0])

        # label
        labels = read_label(pjoin(label_path, f))
        if len(labels) == 0: return
        write_lines(labels, pjoin(label_2_dir, '%08d.txt' % frame))
        frame_list.append(f.split('.')[0]+'\n')

        # points
        pnp = read_txt(pjoin(points_path, f))
        nf = pjoin(velodyne_dir, f.split('.')[0]+'.bin')
        pnp.tofile(nf)

        # calibration
        cp_calib = pjoin(calib_dir, '%08d.txt' % frame)
        copyfile(calib_sample, cp_calib)

        # image
        im = Image.fromarray(image_sample)
        im.save(pjoin(image_2_dir, '%08d.png' % frame))

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pnp[:, :3])
        # o3d.visualization.draw_geometries([pcd])
        # break

    threads = []
    for f in pfiles:
        x = threading.Thread(target=make_data, args=(f,))
        x.start()
        threads.append(x)

    for thr in threads:
        thr.join()
    
    random.shuffle(frame_list)
    train_num = int(len(frame_list) * 0.8)
    train_list = frame_list[:train_num]
    val_list = frame_list[train_num:]
    train_list.sort()
    val_list.sort()
    write_lines(train_list, pjoin(sets_dir, 'train.txt'))
    write_lines(val_list, pjoin(sets_dir, 'val.txt'))