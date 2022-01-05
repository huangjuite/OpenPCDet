import os
import open3d as o3d
import numpy as np
import pandas as pd
import torch

label_map = {'car': 'Car', 'truck': 'Car', 'bus': 'Car',
             'bimo': 'Cyclist', 'pedestrian': 'Pedestrian'}


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(
        corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


range_limit = [0, -40, -3, 70, 40, 1]


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
    dets = []
    for l in la.iterrows():
        if l[1][1] not in label_map.keys():
            continue
        ctype = label_map[l[1][1]]
        pos = l[1][2:5].to_numpy().astype(np.float32)
        length = float(l[1][5])
        width = float(l[1][6])
        height = float(l[1][7])
        ry = float(l[1][8])

        if not in_range(range_limit[3], range_limit[0], pos[0]) \
                or not in_range(range_limit[4], range_limit[1], pos[1]) \
                or not in_range(range_limit[5], range_limit[2], pos[2]):
            continue

        det_lines.append('%s %d %d %d %d %d %d %d %f %f %f %f %f %f %f\n' % (
            ctype, 0, 0, 0, 1, 2, 3, 4, height, width, length, pos[0], pos[1], pos[2], ry))
        dets.append([pos[0], pos[1], pos[2], length, width, height, ry])

    return dets

def read_kitti_label(f):
    la = pd.read_csv(f, sep=" ", header=None)
    dets = []
    for l in la.iterrows():
        l = l[1]
        pos = l[11:14].to_numpy().astype(np.float32)
        length = float(l[10])
        width = float(l[9])
        height = float(l[8])
        ry = float(l[14])
        dets.append([pos[0], pos[1], pos[2], length, width, height, ry])

    return dets
        

if __name__ == '__main__':
    # points = read_txt('point_216.txt')
    # labels = np.array(read_label('anno_216.txt'))

    points = np.fromfile('00000216.bin', dtype=np.float32).reshape(-1, 4)
    labels = np.array(read_kitti_label('00000216.txt'))
    
    boxes = boxes_to_corners_3d(labels)

    geometry = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    geometry.append(pcd)

    for i, bo in enumerate(boxes):
        boxP = o3d.utility.Vector3dVector(bo)
        o3dBox = o3d.geometry.OrientedBoundingBox().create_from_points(boxP)
        # o3dBox.color = colors[labels[i]-1]
        geometry.append(o3dBox)

    o3d.visualization.draw_geometries(geometry)
