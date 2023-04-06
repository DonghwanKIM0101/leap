from collections import defaultdict

import os
import numpy as np
import trimesh
import torch
import tqdm
import open3d as o3d

import sys
sys.path.insert(0, '/home/donghwan/occupancy_networks')
from im2mesh.utils import libmcubes
from im2mesh.utils.libmise import MISE
from im2mesh.utils.libsimplify import simplify_mesh
from im2mesh.common import make_3d_grid

from leap.modules import LEAPModel, INVLBS, FWDLBS
from leap.tools.libmesh import check_mesh_contains


def xyz_to_xyz1(xyz):
    """ Convert xyz vectors from [BS, ..., 3] to [BS, ..., 4] for matrix multiplication
    """
    ones = torch.ones([*xyz.shape[:-1], 1], device=xyz.device)
    return torch.cat([xyz, ones], dim=-1)

    
class BaseTrainer:
    """ Base trainers class.

    Args:
        model (torch.nn.Module): Occupancy Network model
        optimizer (torch.optim.Optimizer): pytorch optimizer object
        cfg (dict): configuration
    """

    def __init__(self, model, optimizer, cfg):
        self.model = model
        self.optimizer = optimizer
        self.device = cfg['device']

    def evaluate(self, val_loader):
        """ Performs an evaluation.
        Args:
            val_loader (torch.DataLoader): pytorch dataloader
        """
        eval_list = defaultdict(list)

        for data in tqdm.tqdm(val_loader):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def _train_mode(self):
        self.model.train()

    def train_step(self, data):
        self._train_mode()
        self.optimizer.zero_grad()
        loss_dict = self.compute_loss(data)
        loss_dict['total_loss'].backward()
        self.optimizer.step()
        return {k: v.item() for k, v in loss_dict.items()}

    @torch.no_grad()
    def eval_step(self, data):
        """ Performs an evaluation step.

        Args:
            data (dict): datasets dictionary
        """
        self.model.eval()
        eval_loss_dict = self.compute_eval_loss(data)
        return {k: v.item() for k, v in eval_loss_dict.items()}

    def compute_loss(self, *kwargs):
        """ Computes the training loss.

        Args:
            kwargs (dict): datasets dictionary
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute_eval_loss(self, data):
        """ Computes the validation loss.

        Args:
            data (dict): datasets dictionary
        """
        return self.compute_loss(data)


class PAIFTrainer(BaseTrainer):
    def __init__(self, model: LEAPModel, optimizer: torch.optim.Optimizer, cfg: dict):
        super().__init__(model, optimizer, cfg)

        self._eval_lbs_mode()
        self.occ_loss = torch.nn.MSELoss()
        self.corr_loss = torch.nn.MSELoss()

        self.scale = 1
        self.occ = None
        self.gt_occ = None
        self.refined_pts = None
        self.posed_points = None
        self.vis_pts = None
        self.corr_pts = None
        self.gt_corr_pts = None
        self.num_corr_pts = 0


    @torch.no_grad()
    def visualize(self, vis_loader, mesh_dir='mesh'):
        self.model.eval()

        eval_list = defaultdict(list)

        for idx, data in enumerate(tqdm.tqdm(vis_loader)):

            #export_point_cloud(data[1]['right']['gt_mano_joints'][0].cpu().numpy(), 'joints')
            #exit(0)

            if idx == 0:
                can_mesh = self.generate_can_mesh(data)
                can_mesh = simplify_mesh(can_mesh, 2048, 5.)

                #can_mesh = self.can_mesh # WARNING

                can_mesh.export(os.path.join(mesh_dir, 'can_mesh.ply'))
                can_vertices = torch.Tensor(can_mesh.vertices).unsqueeze(0).to(device=self.device)
                can_faces = can_mesh.faces

            data['can_points'] = can_vertices.repeat_interleave(15, 0)

            self.forward_pass(data)

            eval_step_dict = {
                'iou': self.compute_iou(self.occ >= 0.5, self.gt_occ >= 0.5).mean(),
                'corr': self.corr_loss(self.corr_pts, self.gt_corr_pts),
                'corr_before_shape': self.corr_loss(self.posed_points, self.gt_corr_pts)
                }
            for k, v in eval_step_dict.items():
                eval_list[k].append(v.item())


            posed_vertices = self.vis_pts
            camera_params = {}
            camera_params['R'] = data['R'].to(device=self.device)
            camera_params['T'] = data['T'].to(device=self.device)
            posed_vertices = torch.bmm(posed_vertices.float(), camera_params['R'])
            posed_vertices += data['root_xyz'].to(device=self.device).unsqueeze(1)
            posed_vertices += data['joints_root'].to(device=self.device)
            posed_vertices = torch.bmm(posed_vertices.float(), camera_params['R'].transpose(1,2))
            posed_vertices += camera_params['T'].double().unsqueeze(1)


            for b_idx in range(15):

                out_dir = data['out_dir'][b_idx] 
                out_fname = data['out_file'][b_idx]

                posed_vertex = posed_vertices[b_idx].detach().cpu().numpy()
                posed_mesh = trimesh.Trimesh(posed_vertex, can_faces, process=False)

                trimesh.repair.fix_normals(posed_mesh)

                if not os.path.exists(os.path.join(mesh_dir, out_dir)):
                    os.system('mkdir -p ' + os.path.join(mesh_dir, out_dir))
                
                posed_mesh.export(os.path.join(mesh_dir, out_dir, f'{out_fname}.ply'))

                norm_posed_vertices = self.vis_pts[b_idx].detach().cpu().numpy()
                norm_posed_mesh = trimesh.Trimesh(norm_posed_vertices, can_faces, process=False)
                norm_posed_mesh.export(os.path.join(mesh_dir, out_dir, f'{out_fname}_norm.ply'))

                # gt_posed_vertices = self.gt_corr_pts[b_idx].detach().cpu().numpy()
                # gt_posed_faces = data['can_faces'][b_idx].detach().cpu().numpy()
                # gt_posed_mesh = trimesh.Trimesh(gt_posed_vertices, gt_posed_faces, process=False)
                # gt_posed_mesh.export(os.path.join(mesh_dir, out_dir, f'{out_fname}_gt.ply'))

                posed_pts_name = os.path.join(mesh_dir, out_dir, f'{out_fname}_before_shape.ply')
                posed_points = self.posed_points[b_idx].detach().cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(posed_points) 
                o3d.io.write_point_cloud(posed_pts_name , pcd)

                shaped_pts_name = os.path.join(mesh_dir, out_dir, f'{out_fname}_shape.ply')
                shaped_points = self.shaped_points[b_idx].detach().cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(shaped_points) 
                o3d.io.write_point_cloud(shaped_pts_name , pcd)

                gt_pts_name = os.path.join(mesh_dir, out_dir, f'{out_fname}_gt_pts.ply')
                gt_posed_pts = self.gt_corr_pts[b_idx].detach().cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(gt_posed_pts) 
                o3d.io.write_point_cloud(gt_pts_name , pcd)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        exit()
        return eval_dict


    def _eval_lbs_mode(self):
        self.model.inv_lbs.require_grad = False
        self.model.inv_lbs.eval()

        self.model.fwd_lbs.require_grad = False
        self.model.fwd_lbs.eval()


    def _train_mode(self):
        self.model.train()
        self._eval_lbs_mode()


    def forward_pass(self, data, input_can_points=None):
        img = data['img'].to(device=self.device)
        B = img.shape[0]

        can_vertices = data['can_vertices'].to(device=self.device)
        self.num_corr_pts = can_vertices.shape[1]
        can_faces = data['can_faces'].to(device=self.device)
        can_joint_root = data['can_joint_root'].to(device=self.device)[0]

        if input_can_points is None:
            orig_can_points = data['can_points'].to(device=self.device)
            can_eval = False
            gt_occ = data['can_occ'].to(device=self.device)
        else:
            orig_can_points = input_can_points
            can_eval = True

        # gt_clothed_vertices = data['gt_clothed_vertices'].to(device=self.device)
        # gt_clothed_faces = data['gt_clothed_faces'][0][:data['lengths'][0]]

        can_rel_joints = data['can_rel_joints'].to(device=self.device)
        rel_joint_root = data['joints_root'].to(device=self.device)
        can_pose = data['can_pose'].to(device=self.device)
        fwd_transformation = data['fwd_transformation'].to(device=self.device)
        root_rot_mat = data['root_rot_mat'].to(device=self.device)
        root_xyz = data['root_xyz'].to(device=self.device)
        camera_params = {}
        camera_params['R'] = data['R'].to(device=self.device)
        camera_params['T'] = data['T'].to(device=self.device)
        camera_params['K'] = data['K'].to(device=self.device)

        can_points = torch.cat((can_vertices, orig_can_points), 1)

        with torch.no_grad():
            self.fwd_points_weights = self.model.fwd_lbs(can_points, can_vertices)

            posed_points = self._can2posed_points(can_points, self.fwd_points_weights, fwd_transformation)

        self.model.shape_net(img)

        posed_points -= can_joint_root
        self.posed_points, self.pred_dist = self.model.shape_net.query(posed_points, can_points, root_xyz, rel_joint_root, self.fwd_points_weights, root_rot_mat, camera_params, fixed=True, scale=self.scale)

        self.refined_pts = -self.pred_dist + self.posed_points

        occ_points = self.refined_pts

        if can_eval:
            inv_occ_points = can_points
        else:
            inv_occ_points = occ_points + self.pred_dist

            inv_occ_points = torch.bmm(inv_occ_points.float(), camera_params['R']) 
            inv_occ_points = torch.bmm(inv_occ_points, root_rot_mat)[:, :, :3]
            inv_occ_points += can_joint_root
            inv_occ_points = self._posed2can_points(inv_occ_points, self.fwd_points_weights, fwd_transformation)

        occupancy = torch.sigmoid(self.model.leap_occupancy_decoder(
            can_points=inv_occ_points, point_weights=self.fwd_points_weights, rot_mats=can_pose, rel_joints=can_rel_joints))
        
        if (can_eval):
            gt_occ = torch.zeros((inv_occ_points.shape[0], inv_occ_points.shape[1])).to(device=self.device)
            can_mesh = trimesh.Trimesh(can_vertices[0].detach().cpu().numpy(), can_faces[0].detach().cpu().numpy(), process=False)

            for b_idx in range(inv_occ_points.shape[0]):
                occ_points_curr = inv_occ_points[b_idx].clone()
                occ_points_curr = occ_points_curr.detach().cpu().numpy()
                gt_occ_curr = check_mesh_contains(can_mesh, occ_points_curr).astype(np.float32)
                gt_occ[b_idx] = torch.Tensor(gt_occ_curr)
            
            gt_occ = gt_occ[:, self.num_corr_pts:]
        
        self.occ = occupancy[:, self.num_corr_pts:]
        self.gt_occ = gt_occ
        self.corr_pts = self.refined_pts[:, :self.num_corr_pts, :]
        self.gt_corr_pts = data['pseudo_gt_corr'].to(device=self.device)
        self.vis_pts = self.refined_pts[:, self.num_corr_pts:, :]
        self.shaped_points = self.refined_pts[:, :self.num_corr_pts, :]
        self.posed_points = self.posed_points[:, :self.num_corr_pts, :]

    
    @torch.no_grad()
    def compute_eval_loss(self, data):

        self.forward_pass(data)

        return {
            'iou': self.compute_iou(self.occ >= 0.5, self.gt_occ >= 0.5).mean(),
            'corr': self.corr_loss(self.corr_pts, self.gt_corr_pts),
            'corr_before_shape': self.corr_loss(self.posed_points, self.gt_corr_pts)
        }

    
    def compute_loss(self, data):
        
        self._train_mode()
        self.forward_pass(data)

        loss_dict = {
            'occ_loss': self.occ_loss(self.occ, self.gt_occ),
            'corr_loss': self.corr_loss(self.corr_pts, self.gt_corr_pts),
            'corr_loss_before_shape': self.corr_loss(self.posed_points, self.gt_corr_pts)
        }
        loss_dict['total_loss'] = 10 * loss_dict['corr_loss'] + loss_dict['occ_loss']
        
        return loss_dict

    
    @staticmethod
    def compute_iou(occ1, occ2):
        """ Computes the Intersection over Union (IoU) value for two sets of
        occupancy values.

        Args:
            occ1 (tensor): first set of occupancy values
            occ2 (tensor): second set of occupancy values
        """
        # Also works for 1-dimensional data
        if len(occ1.shape) >= 2:
            occ1 = occ1.reshape(occ1.shape[0], -1)
        if len(occ2.shape) >= 2:
            occ2 = occ2.reshape(occ2.shape[0], -1)

        # Convert to boolean values
        occ1 = (occ1 >= 0.5)
        occ2 = (occ2 >= 0.5)

        # Compute IOU
        area_union = (occ1 | occ2).float().sum(axis=-1)
        area_intersect = (occ1 & occ2).float().sum(axis=-1)

        iou = (area_intersect / area_union)

        return iou


    def _can2posed_points(self, points, point_weights, fwd_transformation):
        """
        Args:
            points: B x T x 3
            point_weights: B x T x K
            fwd_transformation: B x K x 4 x 4

        Returns:
            canonical points: B x T x 3
        """
        B, T, K = point_weights.shape
        point_weights = point_weights.view(B * T, 1, K)  # B*T x 1 x K

        fwd_transformation = fwd_transformation.unsqueeze(1).repeat(1, T, 1, 1, 1)  # B X K x 4 x 4 -> B x T x K x 4 x 4
        fwd_transformation = fwd_transformation.view(B * T, K, -1)  # B*T x K x 16
        trans = torch.bmm(point_weights, fwd_transformation).view(B * T, 4, 4)

        points = torch.cat([points, torch.ones(B, T, 1, device=points.device)], dim=-1).view(B * T, 4, 1)
        posed_points = torch.bmm(trans, points)[:, :3, 0].view(B, T, 3)

        return posed_points


    def _posed2can_points(self, points, point_weights, fwd_transformation):
        """
        Args:
            points: B x T x 3
            point_weights: B x T x K
            fwd_transformation: B x K x 4 x 4

        Returns:
            canonical points: B x T x 3
        """
        B, T, K = point_weights.shape
        point_weights = point_weights.view(B * T, 1, K)  # B*T x 1 x K

        fwd_transformation = fwd_transformation.unsqueeze(1).repeat(1, T, 1, 1, 1)  # B X K x 4 x 4 -> B x T x K x 4 x 4
        fwd_transformation = fwd_transformation.view(B * T, K, -1)  # B*T x K x 16
        back_trans = torch.bmm(point_weights, fwd_transformation).view(B * T, 4, 4)
        back_trans = torch.inverse(back_trans)

        points = torch.cat([points, torch.ones(B, T, 1, device=points.device)], dim=-1).view(B * T, 4, 1)
        can_points = torch.bmm(back_trans, points)[:, :3, 0].view(B, T, 3)

        return can_points


    @torch.no_grad()
    def eval_points(self, data, pts, pts_batch_size=20000):
        p_split = torch.split(pts, pts_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            pi = pi.repeat_interleave(15, 0)

            self.forward_pass(data, input_can_points=pi)
            occ_hat = self.occ[0]

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat


    @torch.no_grad()
    def extract_mesh(self, occ_hat, threshold=0.5, padding=4):

        n_x, n_y, n_z = occ_hat.shape
        box_size = 0.1 + padding

        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)

        batch_size = 1
        shape = occ_hat.shape

        #voxels_out = (occ_hat >= threshold)
        #vis.visualize_voxels(
        #    voxels_out, os.path.join('debug', f'vox.png'))

        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)

        vertices -= 0.5
        vertices -= 1

        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        mesh = trimesh.Trimesh(vertices, triangles,
                               process=False)

        return mesh
  

    @torch.no_grad()
    def generate_can_mesh(self, data, resolution0=256, upsampling_steps=0, threshold=0.5, padding=4):

        box_size = 0.1 + padding

        mesh_extractor = MISE(
            resolution0, upsampling_steps, threshold)

        points = mesh_extractor.query()

        if upsampling_steps == 0:
            nx = resolution0
            pointsf = box_size * make_3d_grid(
                    (-0.5,)*3, (0.5,)*3, (nx,)*3)
            values = self.eval_points(data, pointsf).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)

        else:
            while points.shape[0] != 0:
                pointsf = torch.FloatTensor(points).to(self.device)

                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)

                values = self.eval_points(
                    data, pointsf).cpu().numpy()

                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        mesh = self.extract_mesh(value_grid)

        return mesh