from collections import defaultdict

import numpy as np
import trimesh
import torch
import tqdm

from leap.modules import LEAPModel, INVLBS, FWDLBS
from leap.tools.libmesh import check_mesh_contains


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
        self.loss = torch.nn.MSELoss()

        self.scale = 1
        self.occ = None
        self.gt_occ = None
        self.refined_pts = None


    def _eval_lbs_mode(self):
        self.model.inv_lbs.require_grad = False
        self.model.inv_lbs.eval()

        self.model.fwd_lbs.require_grad = False
        self.model.fwd_lbs.eval()


    def _train_mode(self):
        self.model.train()
        self._eval_lbs_mode()


    def forward_pass(self, data):

        img, pose_dict = data
        img = img.to(device=self.device)

        can_vert = pose_dict['can_vertices'].to(device=self.device)
        
        can_points = pose_dict['can_points'].to(device=self.device)

        can_rel_joints = pose_dict['can_rel_joints'].to(device=self.device)
        can_pose = pose_dict['can_pose'].to(device=self.device)
        fwd_transformation = pose_dict['fwd_transformation'].to(device=self.device)
        root_rot_mat = pose_dict['root_rot_mat'].to(device=self.device)
        camera_params = pose_dict['camera_params']

        gt_clothed_verts = pose_dict['gt_clothed_verts'].to(device=self.device)
        gt_clothed_faces = pose_dict['gt_clothed_faces'].to(device=self.device)
        
        # can_points = torch.cat((can_vert, can_points), 1)

        for key in camera_params.keys():
            if key not in ['img_path']:
                camera_params[key] = camera_params[key].to(device=self.device)

        with torch.no_grad():
            self.fwd_points_weights = self.model.fwd_lbs(can_points, can_vert)

            for idx in range(can_points.shape[0]):
                can_root = pose_dict['can_joint_root'][idx].cuda()
                can_points[idx] -= can_root

            posed_points = self._can2posed_points(can_points, self.fwd_point_weights, fwd_transformation)
        
        self.model.shape_net(img)

        posed_points, self.pred_dist = self.model.shape_net.query(posed_points, can_points, self.fwd_point_weights, root_rot_mat, camera_params, fixed=True, scale=self.scale)

        self.posed_points = posed_points

        self.refined_pts = -self.pred_dist + posed_points

        occ_points = self.refined_pts

        inv_occ_points = occ_points + self.pred_dist

        inv_occ_points = torch.bmm(inv_occ_points.float(), camera_params['R']) 
        inv_occ_points = torch.bmm(xyz_to_xyz1(inv_occ_points), root_rot_mat.transpose(1,2))[:, :, :3]

        inv_occ_points = self._posed2can_points(inv_occ_points, self.fwd_point_weights, fwd_transformation)
        
        for idx in range(inv_occ_points.shape[0]):
            can_root = pose_dict['can_joint_root'][idx].cuda()
            inv_occ_points[idx] += can_root

        occupancy = torch.sigmoid(self.model.leap_occupancy_decoder(
            can_points=inv_occ_points, point_weights=self.fwd_point_weights, rot_mats=can_pose, rel_joints=can_rel_joints))
        
        gt_occ = torch.zeros((inv_occ_points.shape[0], inv_occ_points.shape[1])).to(device=self.device)
        can_clothed_verts = self._posed2can_points(gt_clothed_verts[0], self.fwd_point_weights, fwd_transformation)

        can_mesh = trimesh.Trimesh(can_clothed_verts.detach().cpu().numpy(), gt_clothed_faces[0].detach().cpu().numpy(), process=False)

        for b_idx in range(inv_occ_points.shape[0]):
            occ_points_curr = inv_occ_points[b_idx].clone()
            occ_points_curr = occ_points_curr.detach().cpu().numpy()
            gt_occ_curr = check_mesh_contains(can_mesh, occ_points_curr).astype(np.float32)
            gt_occ[b_idx] = torch.Tensor(gt_occ_curr)
        
        self.occ = occupancy
        self.gt_occ = gt_occ

    
    @torch.no_grad()
    def compute_eval_loss(self, data):

        self.forward_pass(data)

        return {
            'iou': self.compute_iou(self.occ >= 0.5, self.gt_occ >= 0.5).mean()
        }

    
    def compute_loss(self, data):
        
        self._train_mode()
        self.forward_pass(data)

        loss_dict = {
            'occ_loss': self.loss(self.occ, self.gt_occ)
        }

    
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