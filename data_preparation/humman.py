import argparse
from glob import glob
import json
import os
from os.path import basename, join, splitext
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation

import numpy as np
import torch

from leap.leap_body_model import LEAPBodyModel
from training_code.datasets.mesh_utils import *


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
                body_pose = smpl_params['body_pose']
                betas = smpl_params['betas']
                transl = smpl_params['transl']

                mesh_path = join(args.src_dataset_path, seq_dir, 'reduced_meshes', f"{frame_format}.obj")
                image_path = join(args.src_dataset_path, seq_dir, 'kinect_color', 'kinect_000', f"{frame_format}.png")
                mask_path = join(args.src_dataset_path, seq_dir, 'kinect_mask', 'kinect_000', f"{frame_format}.png")

                if (not os.path.isfile(mesh_path)):
                    continue

                camera_path = join(args.src_dataset_path, seq_dir, 'cameras.json')
                with open(camera_path, 'r') as f:
                    cameras = json.load(f)
                camera_params = cameras['kinect_color_000']  # e.g., Kinect ID = 0
                K, R, T = camera_params['K'], camera_params['R'], camera_params['T']

                b_size = 1

                # print(body_pose.shape)
                leap_body_model = LEAPBodyModel(bm_path=bm_path, num_betas=betas.shape[0])
                leap_body_model.set_parameters(
                    betas=torch.Tensor(betas).view(1,-1),
                    # pose_body=torch.Tensor(body_pose[3:66]).view(1,-1)  # 21 joints
                    pose_body=torch.Tensor(body_pose[:63]).view(1,-1)  # 21 joints
                )
                leap_body_model.forward_parametric_model()

                verts, faces = obj_loader(mesh_path)

                rotation_matrix = Rotation.from_rotvec(global_orient).as_matrix()
                # rotation_matrix = np.concatenate((rotation_matrix, np.expand_dims(transl, axis=1)),1)


                to_save = {
                    'can_vertices': leap_body_model.can_vert,
                    'posed_vertices': leap_body_model.posed_vert,
                    'pose_mat': leap_body_model.pose_rot_mat,
                    'root_rot_mat': [rotation_matrix],
                    'root_xyz': [transl],
                    'clothed_vertices': [verts],
                    'clothed_faces': [faces],
                    'rel_joints': leap_body_model.rel_joints,
                    'fwd_transformation': leap_body_model.fwd_transformation,
                    'mesh_path': [mesh_path],
                    'image_path': [image_path],
                    'mask_path': [mask_path],
                }
                to_save['camera_params'] = {}
                to_save['camera_params']['K'] = K
                to_save['camera_params']['R'] = R
                to_save['camera_params']['T'] = T
                to_save['camera_params'] = [to_save['camera_params']]

                dir_path = join(args.dst_dataset_path, split)
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                # print(f'Saving:\t{dir_path}')
                for b_ind in range(b_size):
                    with open(join(dir_path, '%03d_%03d.npz'%(seq_idx, frame_idx)), 'wb') as file:
                        np.savez(file, **{key: to_np(val[b_ind]) for key, val in to_save.items()})


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
