import os
import random
import numpy as np
import pandas as pd
import open3d as o3d
from os.path import join as pjoin
from tqdm import tqdm
from tqdm.cli import main

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
calib_sample = '/media/cruw/My Passport/livox_sim/simu_data/kitti_sample/calib.txt'


def read_txt(f):
    with open(f, 'r') as h:
        pnp = pd.read_csv(f, sep=",", header=None).to_numpy().astype(
            np.float32)[:, :4]
        pnp[:, 3] = 0
        return pnp


if __name__ == '__main__':
    main_path = '/media/cruw/My Passport/livox_sim/simu_data/points'
    save_dir = pjoin(main_path, '..', 'kitti_sim')
    os.makedirs(save_dir, exist_ok=True)
    pfiles = os.listdir(main_path)
    pfiles.sort()


    for f in tqdm(pfiles):
        pnp = read_txt(pjoin(main_path, f))
        nf = pjoin(save_dir, f.split('.')[0]+'.bin')
        pnp.tofile(nf)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pnp[:, :3])
        # o3d.visualization.draw_geometries([pcd])
        break
