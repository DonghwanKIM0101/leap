import torch.nn as nn
import sys
import cv2
import numpy as np
import torchvision
import torch
import torch.nn.functional as F

from .encoders import BaseModule
from .HGFilters import HGFilter


import sys
sys.path.insert(0, '/home/donghwan/leap/leap/modules') 
import geometry

def xyz_to_xyz1(xyz):
    """ Convert xyz vectors from [BS, ..., 3] to [BS, ..., 4] for matrix multiplication
    """
    ones = torch.ones([*xyz.shape[:-1], 1], device=xyz.device)
    return torch.cat([xyz, ones], dim=-1)


def pts_to_img_coord(p, root_xyz, root_joint, root_rot_mat, camera_params):

    img_p = torch.bmm(p, root_rot_mat.transpose(1,2))[:, :, :3]

    trans_img_p = torch.bmm(img_p.float(), camera_params['R'].transpose(1,2)) 

    img_p = img_p + root_xyz.unsqueeze(1)
    img_p = img_p + root_joint
    img_p = torch.bmm(img_p, camera_params['R'].transpose(1,2))
    img_p += camera_params['T'].unsqueeze(1) # this aligns well with world-coordinate mesh 

    img_p = torch.bmm(img_p, camera_params['K'].transpose(1,2))
    
    proj_img_p = torch.zeros((img_p.shape[0], img_p.shape[1], 2)).cuda()
    for i in range(img_p.shape[0]):
        proj_img_p[i] = img_p[i, :, :2] / img_p[i, :, 2:] 

    '''
    import cv2
    import torchvision.transforms as transforms
    trans = transforms.ToPILImage()
    for i in range(15):
        img = np.array(trans(image[i].detach().cpu()))
        sub_img_p = proj_img_p[i] 

        for idx in range(sub_img_p.shape[0]):
            pt = sub_img_p[idx]
            if pt.min() < 0 or pt.max() > 255: continue
            img[int(pt[1]), int(pt[0])] = [255, 0, 255]

        cv2.imwrite('debug/check_%d.jpg' % i, img)
    exit()
    '''
    norm_z = torch.ones(img_p[:, :, 2].shape).cuda() - img_p[:, :, 2]
    return trans_img_p, proj_img_p, norm_z 


# inherit both LEAP and LVD base networks
class NetworkBase(BaseModule):
    def __init__(self):
        super(BaseModule, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            try:
                m.weight.data.normal_(0.0, 0.02)
            except:
                for i in m.children():
                    i.apply(self._weights_init_fn)
                return
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def _get_norm_layer(self, norm_type='batch'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type =='batchnorm2d':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        return norm_layer


class ShapeNet(NetworkBase):

    def __init__(self, input_point_dimensions=3, input_channels=3, pred_dimensions=3):
        super(NetworkBase, self).__init__()

        self._name = 'DeformationNet'

        self.image_filter = HGFilter(3, 2, input_channels, 64, 'group', 'no_down', False)

        input_point_dimensions=3 # Need ot fix the bug

        self.gru = nn.GRU(222, 64, 2, batch_first=True, bidirectional=True)

        self.fc1 = nn.utils.weight_norm(nn.Conv1d(350, 256, kernel_size=1, bias=True))
        self.fc2 = nn.utils.weight_norm(nn.Conv1d(256, 512, kernel_size=1, bias=True))
        self.fc3 = nn.utils.weight_norm(nn.Conv1d(512, pred_dimensions, kernel_size=1, bias=True))

        self.frequencies = [  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.]

    def forward(self, image):
        self.im_feat_list, self.normx = self.image_filter(image)
        # self.img = image
        return


    def query(self, points, can_points, root_xyz, root_joint, lbs_weights, root_rot_mat, camera_params, scale=1, fixed=False):

        # print('check - org query')

        trans_img_p, points, norm_z = pts_to_img_coord(points, root_xyz, root_joint, root_rot_mat, camera_params)
        #xy = (points[:, :, :2] - 128) / 128
        xy = (points - 128) / 128
        xy = torch.cat((xy, norm_z.unsqueeze(-1)), -1)

        can_points *= scale
        if not fixed:
            intermediate_preds_list = torch.cat((xy, can_points, lbs_weights) , 2).transpose(2, 1)
        else:
            intermediate_preds_list = torch.cat((trans_img_p, can_points, lbs_weights) , 2).transpose(2, 1)

        # print(intermediate_preds_list.shape)
        for j, im_feat in enumerate(self.im_feat_list):
            point_local_feat_list = [intermediate_preds_list, geometry.index(im_feat.float(), xy.float())]
            intermediate_preds_list = torch.cat(point_local_feat_list, 1)
        # print(intermediate_preds_list.shape)
        shape_code = intermediate_preds_list.permute(2, 0, 1)
        shape_code, _ = self.gru(shape_code.float())
        shape_code = shape_code.permute(1, 2, 0)

        intermediate_preds_list = torch.cat((shape_code, intermediate_preds_list),1)

        x = F.relu(self.fc1(intermediate_preds_list.float()))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.transpose(1,2)

        return trans_img_p, x


    @classmethod
    def from_cfg(cls, config):
        model = cls()

        return model