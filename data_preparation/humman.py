import argparse
from glob import glob
import json
import os
from os.path import basename, join, splitext
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import open3d as o3d

import numpy as np
import torch

from leap.leap_body_model import LEAPBodyModel
from training_code.datasets.mesh_utils import *

def nearest_vertices(smpl_vertices, cloth_vertices):
    cloth_vertices = torch.Tensor(cloth_vertices).unsqueeze(0).to(device='cuda')
    distances = torch.cdist(smpl_vertices, cloth_vertices)
    _, min_index = torch.min(distances, dim=2)
    idx1 = torch.arange(smpl_vertices.shape[0]).view(-1, 1)
    return cloth_vertices[idx1, min_index]


@torch.no_grad()
def main(args):

    for split in args.splits:
        split_path = join(args.src_dataset_path, f"{split}.txt")
        with open(split_path, 'r') as split_file:
            split_dirs = split_file.read()
            split_dirs = split_dirs.replace('\n', ' ').split()

        for seq_idx, seq_dir in enumerate(split_dirs):
            print(f'Processing {seq_idx+1} out of {len(split_dirs)} seqs...')
            seq_path = join(args.src_dataset_path, seq_dir, 'smpl_params')

            for frame_idx in range(len(glob(join(seq_path, "*.npz")))):
                frame_format = "%06d"%(6*frame_idx)
                smpl_params = np.load(join(seq_path, f"{frame_format}.npz"))
                bm_path = join(args.bm_dir_path, 'SMPL_NEUTRAL.pkl')
                global_orient = smpl_params['global_orient']
                body_pose = torch.Tensor(smpl_params['body_pose']).to(device='cuda')
                betas = torch.Tensor(smpl_params['betas']).to(device='cuda')
                transl = smpl_params['transl']

                mesh_path = join(args.src_dataset_path, seq_dir, 'tight_meshes', f"{frame_format}.obj")

                if (not os.path.isfile(mesh_path)):
                    continue

                camera_path = join(args.src_dataset_path, seq_dir, 'cameras.json')
                with open(camera_path, 'r') as f:
                    cameras = json.load(f)

                b_size = 1

                leap_body_model = LEAPBodyModel(bm_path=bm_path, num_betas=betas.shape[0], device='cuda')
                leap_body_model.set_parameters(
                    betas=betas.view(1,-1),
                    # pose_body=torch.Tensor(body_pose[3:66]).view(1,-1)  # 21 joints
                    pose_body=body_pose[:63].view(1,-1)  # 21 joints
                )
                leap_body_model.forward_parametric_model()

                verts, faces = obj_loader(mesh_path)

                rotation_matrix = Rotation.from_rotvec(global_orient).as_matrix()

                joints_root = leap_body_model.rel_joints[0].cpu().numpy()[0]
                
                verts = verts.T
                verts = verts - joints_root
                verts = verts - np.expand_dims(transl, 1)
                verts = np.matmul(rotation_matrix.T, verts)
                verts = verts + joints_root
                verts = verts.T

                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(leap_body_model.can_vert[0].cpu().numpy()) 
                # o3d.io.write_point_cloud('debug/can_mesh.ply' , pcd)

                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(leap_body_model.posed_vert[0].cpu().numpy()) 
                # o3d.io.write_point_cloud('debug/pose_mesh.ply' , pcd)

                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(verts) 
                # o3d.io.write_point_cloud('debug/psuedo_mesh.ply' , pcd)

                # exit()

                to_save = {
                    'can_vertices': leap_body_model.can_vert[0],
                    'posed_vertices': leap_body_model.posed_vert[0],
                    'pose_mat': leap_body_model.pose_rot_mat[0],
                    'root_rot_mat': rotation_matrix,
                    'root_xyz': transl,
                    'clothed_vertices': verts,
                    'clothed_faces': faces,
                    'pseudo_gt_corr': nearest_vertices(leap_body_model.posed_vert, verts)[0],
                    'rel_joints': leap_body_model.rel_joints[0],
                    'fwd_transformation': leap_body_model.fwd_transformation[0],
                    'mesh_path': mesh_path,
                }

                dir_path = join(args.dst_dataset_path, split, seq_dir)
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                with open(join(dir_path, f"{frame_format}.npz"), 'wb') as file:
                    np.savez(file, **{key: to_np(val) for key, val in to_save.items()})

                for cam_idx in range(10):
                    cam_dir = 'kinect_%03d' % cam_idx
                    image_path = join(args.src_dataset_path, seq_dir, 'kinect_color', cam_dir, f"{frame_format}.png")
                    mask_path = join(args.src_dataset_path, seq_dir, 'kinect_mask', cam_dir, f"{frame_format}.png")

                    sub_dir = join(seq_dir, cam_dir)

                    camera_params = cameras[f'kinect_color_00{cam_idx}']
                    K, R, T = camera_params['K'], camera_params['R'], camera_params['T']

                    to_save_multiview = {
                        'image_path': image_path,
                        'mask_path': mask_path,
                        'out_dir': sub_dir,
                        'out_file': frame_format
                    }
                    to_save_multiview['camera_params'] = {}
                    to_save_multiview['camera_params']['K'] = K
                    to_save_multiview['camera_params']['R'] = R
                    to_save_multiview['camera_params']['T'] = T

                    dir_path = join(args.dst_dataset_path, split, seq_dir, cam_dir)
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    with open(join(dir_path, f"{frame_format}.npz"), 'wb') as file:
                        np.savez(file, **{key: to_np(val) for key, val in to_save_multiview.items()})


def to_np(variable):
    if torch.is_tensor(variable):
        variable = variable.detach().cpu().numpy()

    return variable


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess HuMMan dataset.')
    parser.add_argument('--src_dataset_path', type=str, required=True,
                        help='Path to HuMMan dataset.')
    parser.add_argument('--dst_dataset_path', type=str, required=True,
                        help='Directory path to store preprocessed dataset.')
    parser.add_argument('--splits', type=list, required=False, default=['train', 'test'],
                        help='Split of HuMMan to use, separated by comma.')
    parser.add_argument('--bm_dir_path', type=str, required=True,
                        help='Path to SMPL model')

    main(parser.parse_args())
