from collections import defaultdict

import numpy as np
import trimesh
import torch
import tqdm

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
        img = data['img'].to(device=self.device)
        B = img.shape[0]

        can_vertices = data['can_vertices'].to(device=self.device)
        can_faces = data['can_faces'].to(device=self.device)
        can_joint_root = data['can_joint_root'].to(device=self.device)[0]
        orig_can_points = data['can_points'].to(device=self.device)
        
        # gt_clothed_vertices = data['gt_clothed_vertices'].to(device=self.device)
        # gt_clothed_faces = data['gt_clothed_faces'][0][:data['lengths'][0]]

        can_rel_joints = data['can_rel_joints'].to(device=self.device)
        rel_joint_root = data['rel_joints'][:, :1, :, 0].to(device=self.device)
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
        posed_points, self.pred_dist = self.model.shape_net.query(posed_points, can_points, root_xyz, rel_joint_root, self.fwd_points_weights, root_rot_mat, camera_params, fixed=True, scale=self.scale)

        self.posed_points = posed_points

        self.refined_pts = -self.pred_dist + posed_points

        occ_points = self.refined_pts

        inv_occ_points = occ_points + self.pred_dist

        inv_occ_points = torch.bmm(inv_occ_points.float(), camera_params['R']) 
        inv_occ_points = torch.bmm(inv_occ_points, root_rot_mat.transpose(1,2))[:, :, :3]
        posed_points += can_joint_root
        inv_occ_points = self._posed2can_points(inv_occ_points, self.fwd_points_weights, fwd_transformation)

        occupancy = torch.sigmoid(self.model.leap_occupancy_decoder(
            can_points=inv_occ_points, point_weights=self.fwd_points_weights, rot_mats=can_pose, rel_joints=can_rel_joints))
        
        gt_occ = torch.zeros((inv_occ_points.shape[0], inv_occ_points.shape[1])).to(device=self.device)
        can_mesh = trimesh.Trimesh(can_vertices[0].detach().cpu().numpy(), can_faces[0].detach().cpu().numpy(), process=False)

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
        loss_dict['total_loss'] = loss_dict['occ_loss']
        
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