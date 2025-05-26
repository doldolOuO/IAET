import torch
import torch.utils.data
import torch.nn as nn
from models.model_utils import InterlacedTransformer, ViewGuidedUpLayer
from models.utils import MLP_Res, PointNet_SA_Module_KNN
from torchvision import models


class PcEncoder(nn.Module):
    def __init__(self, dim_feat=256, num_points=256):
        super(PcEncoder, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512,
                                                  16,
                                                  3,
                                                  [64, 128],
                                                  group_all=False,
                                                  if_bn=False,
                                                  if_idx=True)
        self.sa_module_2 = PointNet_SA_Module_KNN(num_points,
                                                  16,
                                                  128,
                                                  [128, dim_feat],
                                                  group_all=False,
                                                  if_bn=False,
                                                  if_idx=True)

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, 3, N)
        Returns:
            l2_points: (B, dim_feat, num_points)
        """
        l0_xyz = point_cloud
        l0_points = point_cloud
        l1_xyz, l1_points, _ = self.sa_module_1(l0_xyz, l0_points)
        l2_xyz, l2_points, _ = self.sa_module_2(l1_xyz, l1_points)
        return l2_points


class ImEncoder(nn.Module):
    def __init__(self, dim_feat=256):
        super().__init__()
        self.dim_feat = dim_feat
        base = models.resnet18(pretrained=False)
        self.base = nn.Sequential(*list(base.children())[:-3])

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), self.dim_feat, -1)
        return x


class CoarseDecoder(nn.Module):
    def __init__(self, dim_feat=256, num_points=256):
        super(CoarseDecoder, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat,
                                     128,
                                     num_points,
                                     bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128,
                             hidden_dim=128,
                             out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128,
                             hidden_dim=64,
                             out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128,
                             hidden_dim=128,
                             out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128,
                      64,
                      1),
            nn.ReLU(),
            nn.Conv1d(64,
                      3,
                      1)
        )

    def forward(self, feat):
        """
        Args:
            feat: (B, dim_feat, N)
        Returns:
            completion: (B, 3, num_points)
        """
        f_g = torch.max(feat, -1, keepdim=True)[0]
        x1 = self.ps(f_g)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1,
                                   f_g.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2,
                                   f_g.repeat((1, 1, x2.size(2)))], 1))
        completion = self.mlp_4(x3)
        return completion


class Decoder(nn.Module):
    def __init__(self, dim_feat, num_points, up_factors):
        super(Decoder, self).__init__()
        self.coarse_decoder = CoarseDecoder(dim_feat, num_points)
        uppers = []
        for _, factor in enumerate(up_factors):
            uppers.append(ViewGuidedUpLayer(dim_feat, factor))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, fused_pc_token, fused_im_token):
        arr_pcd = []
        pcd = self.coarse_decoder(fused_pc_token)
        arr_pcd.append(pcd.permute(0, 2, 1).contiguous())
        K_prev = None
        for upper in self.uppers:
            pcd, K_prev = upper(pcd, fused_pc_token, fused_im_token, K_prev)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())
        return arr_pcd


class IAET(nn.Module):
    def __init__(self, dim_feat=256, num_points=256, up_factors=(2, 2, 2), num_blocks=4):
        super().__init__()
        self.pc_encoder = PcEncoder(dim_feat, num_points)
        self.im_encoder = ImEncoder(dim_feat)
        self.fusion_module = InterlacedTransformer(dim_feat,
                                                   dim_feat,
                                                   dim_feat,
                                                   num_blocks)
        self.decoder = Decoder(dim_feat, num_points, up_factors)

    def forward(self, pc, image):
        # Encoding Stage
        pc_token = self.pc_encoder(pc.permute(0, 2, 1).contiguous())
        im_token = self.im_encoder(image)
        # Fusion Stage
        fused_pc_token, fused_im_token = self.fusion_module(pc_token.permute(0, 2, 1).contiguous(),
                                                            im_token.permute(0, 2, 1).contiguous())
        # Decoding Stage
        pc_completion = self.decoder(fused_pc_token.permute(0, 2, 1).contiguous(),
                                     fused_im_token.permute(0, 2, 1).contiguous())
        return pc_completion


if __name__ == '__main__':
    import numpy as np
    x_part = torch.randn(4, 2048, 3).cuda()
    view = torch.randn(4, 3, 224, 224).cuda()
    model = IAET().cuda()
    out = model(x_part, view)
    for i in range(len(out)):
        print(out[i].shape)