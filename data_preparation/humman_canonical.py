import argparse
from glob import glob
from os.path import basename, join, splitext
from pathlib import Path
import pickle

import numpy as np
import torch

import torchgeometry
from skimage import measure


@torch.no_grad()
def main(args):

    b_size = 1
    humman_mean_betas = torch.Tensor([[-0.67686401, -0.27348559, 0.24219796, 0.25554121, 0.07570673, 0.58607108, 0.09759713, 0.12337149, -0.18763505, 0.23767047]]).repeat(b_size, 1)
    num_betas = 10

    bm_path = join(args.bm_dir_path, 'smpl', 'SMPL_NEUTRAL.pkl')
    with open(bm_path, 'rb') as smpl_file:
        smpl_dict = pickle.load(smpl_file, encoding='latin1')

    kintree_table = smpl_dict['kintree_table'].astype(np.int32)
    kintree_table = torch.tensor(kintree_table, dtype=torch.int32)
    v_template = np.repeat(smpl_dict['v_template'][np.newaxis], b_size, axis=0)
    v_template = torch.tensor(v_template, dtype=torch.float32)
    joint_regressor = smpl_dict['J_regressor']  # V x K
    joint_regressor = torch.tensor(joint_regressor.toarray(), dtype=torch.float32)
    shape_dirs = smpl_dict['shapedirs'][:, :, :num_betas]
    shape_dirs = torch.tensor(np.array(shape_dirs), dtype=torch.float32)
    faces = smpl_dict['f']


    # pose to rot matrices
    full_pose = [torch.tensor(np.zeros((b_size, 24, 3)), dtype=torch.float32, device='cpu')]
    full_pose = torch.cat(full_pose, dim=1)

    pose_rot_mat = torchgeometry.angle_axis_to_rotation_matrix(full_pose.view(-1, 3))[:, :3, :3]
    pose_rot_mat = pose_rot_mat.view(b_size, -1, 3, 3)

    # Compute identity-dependent correctives
    identity_offsets = torch.einsum('bl,mkl->bmk', humman_mean_betas, shape_dirs)
    can_vert = v_template + identity_offsets

    # Regress joint locations
    can_joint_loc = torch.einsum('bik,ji->bjk', v_template + identity_offsets, joint_regressor)

    # Skinning
    B, K = pose_rot_mat.shape[0], can_joint_loc.shape[1]

    parents = kintree_table[0].long()

    joints = torch.unsqueeze(can_joint_loc, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    to_save = {
        'can_vertices': can_vert[0],
        'can_faces': faces,
        'can_joints': can_joint_loc[0],
        'pose_mat': pose_rot_mat[0],
        'rel_joints': rel_joints[0],
    }

    with open(join(args.dst_dataset_path, 'canonical.npz'), 'wb') as file:
        np.savez(file, **{key: to_np(val) for key, val in to_save.items()})


def to_np(variable):
    if torch.is_tensor(variable):
        variable = variable.detach().cpu().numpy()

    return variable


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess HuMMan dataset.')
    parser.add_argument('--dst_dataset_path', type=str, required=True,
                        help='Directory path to store preprocessed dataset.')
    parser.add_argument('--bm_dir_path', type=str, required=True,
                        help='Path to SMPL model')

    main(parser.parse_args())
