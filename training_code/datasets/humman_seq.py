import os
from glob import glob
import json
import pickle
import os.path as osp
from tqdm import tqdm

import numpy as np
import cv2
from PIL import Image
from scipy.spatial import cKDTree as KDTree
from torch.utils import data
from trimesh import Trimesh
from trimesh.remesh import subdivide
from leap.tools.libmesh import check_mesh_contains
from leap.leap_body_model import LEAPBodyModel

import torchvision.transforms as transforms

from training_code.datasets.mesh_utils import *


class TrainSampler():
    """
    Arguments:
    data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, seq_len=17):
        self.data_source = data_source
        self.seq_len = seq_len

    @property
    def num_samples(self):
        return len(self.data_source)

    def __iter__(self):
        n = len(self.data_source)
        seed_n_list = torch.randperm(n).tolist()

        idx_list = []
        for seed_n in tqdm(seed_n_list):

            if self.data_source.return_seq(seed_n) != self.data_source.return_seq(seed_n+self.seq_len):
                continue

            if self.data_source.return_cam(seed_n) != self.data_source.return_cam(seed_n+self.seq_len):
                continue
            
            for i in range(self.seq_len):
                idx_list.append(seed_n + i)

        return iter(idx_list)

    def __len__(self):
        return self.num_samples


class TestSampler():
    """
    Arguments:
    data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, seq_len=17, seq_num=2000):
        self.data_source = data_source
        self.seq_len = seq_len
        self.seq_num = seq_num

        self.n_samples = self.seq_len * self.seq_num

    @property
    def num_samples(self):
        return self.n_samples

    def __iter__(self):

        torch.manual_seed(0)

        n = len(self.data_source)
        seed_n_list = torch.randperm(n).tolist()

        idx_list = []
        for seed_n in tqdm(seed_n_list):

            if self.data_source.return_seq(seed_n) != self.data_source.return_seq(seed_n+self.seq_len):
                continue

            if self.data_source.return_cam(seed_n) != self.data_source.return_cam(seed_n+self.seq_len):
                continue
            
            for i in range(self.seq_len):
                idx_list.append(seed_n + i)

            if len(idx_list) > self.n_samples: 
                break
        return iter(idx_list)

    def __len__(self):
        return self.n_samples


class HuMManSeqDataset(data.Dataset):
    """ HuMMan dataset class for occupancy training. """
    def __init__(self, cfg, split):
        """ Initialization of the the 3D shape dataset.

        Args:
            cfg (dict): dataset configuration
            mode (str): `train`, `val`, or 'test' dataset mode
        """

        # Attributes
        assert split in ['train', 'val', 'test']
        if (split == 'val'):
            split = 'test'
        self.split = split
        self.dataset_folder = cfg['dataset_folder']
        self.split_path = osp.join(self.dataset_folder, f'{self.split}.txt')

        with open(cfg['bm_path'], 'rb') as smpl_file:
            self.faces = pickle.load(smpl_file, encoding='latin1')['f']
        self.faces.dtype = np.int32

        canonical_pose = np.load(osp.join(self.dataset_folder, 'canonical.npz'))
        self.can_vertices = canonical_pose['can_vertices'].astype(np.float32)
        self.can_faces = canonical_pose['can_faces'].astype(int)
        self.can_rel_joints = canonical_pose['rel_joints'].astype(np.float32)
        self.can_pose = canonical_pose['pose_mat'].astype(np.float32)
        self.can_joints = canonical_pose['can_joints'].astype(np.float32)
        self.can_joint_root = self.can_joints[0]

        # Sampling config
        sampling_config = cfg.get('sampling_config', {})
        self.points_uniform_ratio = sampling_config.get('points_uniform_ratio', 0.5)
        self.bbox_padding = sampling_config.get('bbox_padding', 0)
        self.points_padding = sampling_config.get('points_padding', 0.1)
        self.points_sigma = sampling_config.get('points_sigma', 0.01)

        self.n_points_posed = sampling_config.get('n_points_posed', 2048)
        self.n_points_can = sampling_config.get('n_points_can', 2048)

        # Get all models
        self.data_list = self._load_data_files()

        self.normalize_img = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.CenterCrop(256),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
        ])

    def _load_data_files(self):
        data_list = []
        for seq_path in glob(osp.join(self.dataset_folder, self.split, '*')):
            for cam_idx in range(10):
                data_list += sorted(list(glob(osp.join(self.dataset_folder, self.split, seq_path, 'kinect_%03d' % cam_idx, "*.npz"))))

        return data_list

    def __len__(self):
        return len(self.data_list)

    def sample_points(self, mesh, n_points, prefix='', compute_occupancy=False, frame=None):
        # Get extents of model.
        bb_min = np.min(mesh.vertices, axis=0)
        bb_max = np.max(mesh.vertices, axis=0)
        total_size = (bb_max - bb_min).max()

        # Scales all dimensions equally.
        scale = total_size / (1 - self.bbox_padding)
        loc = np.array([(bb_min[0] + bb_max[0]) / 2.,
                        (bb_min[1] + bb_max[1]) / 2.,
                        (bb_min[2] + bb_max[2]) / 2.], dtype=np.float32)

        n_points_uniform = int(n_points * self.points_uniform_ratio)
        n_points_surface = n_points - n_points_uniform

        box_size = 1 + self.points_padding
        points_uniform = np.random.rand(n_points_uniform, 3)
        points_uniform = box_size * (points_uniform - 0.5)
        # Scale points in (padded) unit box back to the original space
        points_uniform *= scale
        points_uniform += loc
        # Sample points around posed-mesh surface
        n_points_surface_cloth = n_points_surface
        points_surface = mesh.sample(n_points_surface_cloth)

        points_surface = points_surface[:n_points_surface_cloth]
        points_surface += np.random.normal(scale=self.points_sigma, size=points_surface.shape)

        # Check occupancy values for sampled points
        query_points = np.vstack([points_uniform, points_surface]).astype(np.float32)

        to_ret = {
            f'{prefix}points': query_points,
            # f'{prefix}loc': loc,
            # f'{prefix}scale': np.asarray(scale),
        }
        if compute_occupancy:
            to_ret[f'{prefix}occ'] = check_mesh_contains(mesh, query_points).astype(np.float32)

        return to_ret

    def return_seq(self, idx):
        try:
            data_path = str(self.data_list[idx])
            sub_dir, _ = osp.split(data_path)
            sub_dir, _ = osp.split(data_path)
            seq_name = osp.split(sub_dir)[-1]
        except:
            seq_name = 'n/a'

        return seq_name

    def return_cam(self, idx):
        try:
            data_path = str(self.data_list[idx])
            sub_dir, _ = osp.split(data_path)
            cam_name = osp.split(sub_dir)[-1]
        except:
            cam_name = 'n/a'

        return cam_name

    def __getitem__(self, idx):
        """ Returns an item of the dataset.

        Args:
            idx (int): ID of datasets point
        """

        sub_dir, file_name = osp.split(self.data_list[idx])
        sub_dir, _ = osp.split(sub_dir)
        data_dict = np.load(osp.join(sub_dir, file_name), allow_pickle=True)
        data_dict_multiview = np.load(self.data_list[idx], allow_pickle=True)
        pose_dict = {}

        color = cv2.imread(str(data_dict_multiview['image_path']))
        mask = cv2.imread(str(data_dict_multiview['mask_path']), cv2.IMREAD_GRAYSCALE)
        color[mask==0] = 0 # gt segmentation
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        img_tensor = self.normalize_img(Image.fromarray(color))

        camera_params = data_dict_multiview['camera_params'].item()
        K, R, T = camera_params['K'], camera_params['R'], camera_params['T']
        K = torch.as_tensor(K).float()
        R = torch.as_tensor(R).float()
        T = torch.as_tensor(T).float()

        # Image is resized and crop.
        K[0,2] -= 420
        K *= 256/1080
        K[2,2] = 1

        can_mesh = Trimesh(self.can_vertices, self.faces, process=False)
        pose_dict.update(self.sample_points(can_mesh, self.n_points_can, prefix='can_', compute_occupancy=True))

        joints_root = data_dict['rel_joints'][0]
        root_rot_mat = data_dict['root_rot_mat']
        root_xyz = data_dict['root_xyz']
        pseudo_gt_corr = data_dict['pseudo_gt_corr'].T
        pseudo_gt_corr = pseudo_gt_corr - joints_root
        pseudo_gt_corr = root_rot_mat @ pseudo_gt_corr
        pseudo_gt_corr = R.cpu().numpy() @ pseudo_gt_corr
        pseudo_gt_corr = pseudo_gt_corr.T

        pose_dict['img'] = img_tensor
        pose_dict['out_dir'] = str(data_dict_multiview['out_dir'])
        pose_dict['out_file'] = str(data_dict_multiview['out_file'])
        pose_dict['fwd_transformation'] = data_dict['fwd_transformation']

        pose_dict['can_vertices'] = self.can_vertices
        pose_dict['can_faces'] = self.can_faces
        pose_dict['can_joint_root'] = torch.Tensor(self.can_joint_root)
        pose_dict['can_rel_joints'] = self.can_rel_joints
        pose_dict['can_pose'] = self.can_pose
        pose_dict['root_rot_mat'] = torch.Tensor(root_rot_mat).float()
        pose_dict['root_xyz'] = torch.Tensor(root_xyz).float()

        pose_dict['pseudo_gt_corr'] = torch.Tensor(pseudo_gt_corr).float()
        pose_dict['joints_root'] = torch.Tensor(joints_root.T).float()

        pose_dict['R'] = R
        pose_dict['T'] = T
        pose_dict['K'] = K

        return pose_dict

