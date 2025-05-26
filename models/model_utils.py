# Author: Chenghao Fang
# date: 2024/10/23
import torch
from torch import nn, einsum
from models.utils import MLP_Res, MLP_CONV, grouping_operation, query_knn

class CrossTransformerBlock(nn.Module):
    def __init__(self,
                 dim=256,
                 dim_hidden=256,
                 num_heads=8,
                 dropout=0.0):
        super().__init__()
        self.LN1 = nn.LayerNorm(dim)
        self.LN2 = nn.LayerNorm(dim)
        self.MHA = nn.MultiheadAttention(dim,
                                         num_heads,
                                         batch_first=True)
        self.LN3 = nn.LayerNorm(dim)
        self.MLP = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim)
        )
        self.drop = nn.Dropout(dropout)


    def forward(self, token1, token2):
        """
        Args:
            token1: (B, N, C)
            token2: (B, M, C)
        Returns:
            token_out: (B, N, C)
        """
        token1 = self.LN1(token1)
        token2 = self.LN2(token2)
        token2_ = self.MHA(token1, token2, token2)[0]
        token1 = self.LN3(token1 + self.drop(token2_))
        token_out = token1 + self.drop(self.MLP(token1))
        return token_out


class InterlacedTransformerBlock(nn.Module):
    def __init__(self,
                 dim=256,
                 dim_hidden=256,
                 num_heads=8):
        super().__init__()
        self.CTB1 = CrossTransformerBlock(dim,
                                          dim_hidden,
                                          num_heads)
        self.CTB2 = CrossTransformerBlock(dim,
                                          dim_hidden,
                                          num_heads)

    def forward(self, pc_token_in, im_token_in):
        """
        Args:
            pc_token_in: (B, N, C)
            im_token_in: (B, M, C)
        Returns:
            pc_token_out: (B, N, C)
            im_token_out: (B, M, C)
        """
        im_token_out = self.CTB1(im_token_in, pc_token_in)
        pc_token_out = self.CTB2(pc_token_in, im_token_out)
        return pc_token_out, im_token_out


class InterlacedTransformer(nn.Module):
    def __init__(self,
                 dim_in=256,
                 dim_out=256,
                 dim_hidden=256,
                 num_heads=8,
                 num_blocks=1):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_embedding = nn.Linear(dim_in, dim_out)
        self.blocks = nn.Sequential(*[
            InterlacedTransformerBlock(
                dim_out,
                dim_hidden,
                num_heads
            )
            for _ in range(num_blocks)])

    def forward(self, pc_token, im_token):
        """
        Args:
            pc_token: (B, N, C)
            im_token: (B, M, C)
        Returns:
            pc_token1: (B, N, C)
            im_token1: (B, M, C)
        """
        pc_token1 = self.token_embedding(pc_token)
        im_token1 = self.token_embedding(im_token)
        for i in range(self.num_blocks):
            pc_token1, im_token1 = self.blocks[i](pc_token1, im_token1)
        return pc_token1, im_token1


class ViewGeoAwareTransformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, num_heads=4):
        super().__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp_view = nn.Sequential(
            nn.Conv2d(dim, dim * num_heads, 1),
            nn.BatchNorm2d(dim * num_heads),
            nn.ReLU(),
            nn.Conv2d(dim * num_heads, dim, 1)
        )

        self.attn_mlp_geo = nn.Sequential(
            nn.Conv2d(dim, dim * num_heads, 1),
            nn.BatchNorm2d(dim * num_heads),
            nn.ReLU(),
            nn.Conv2d(dim * num_heads, dim, 1)
        )

        self.conv_view = nn.Conv1d(dim, in_channel, 1)
        self.conv_geo = nn.Conv1d(dim, in_channel, 1)

    def forward(self, pos, query_im, key_pc, value, include_self=True):
        """
        Args:
            pos: (B, 3, N)
            query_im: (B, in_channel, M)
            key_pc: (B, in_channel, N)
            value: (B, in_channel, N)
            include_self: boolean

        Returns:
            feat_out: (B, in_channel, N)
        """
        # feature projection
        identity = value
        key = self.conv_key(key_pc)
        value = self.conv_value(value)
        # view-guided
        im_norm = torch.norm(query_im, dim=1, keepdim=True)
        pc_norm = torch.norm(key_pc, dim=1, keepdim=True)
        cos_sim = torch.matmul(query_im.permute(0, 2, 1), key_pc) / (im_norm.permute(0, 2, 1) * pc_norm) # B, M, N
        idx_knn_view = torch.argsort(cos_sim, dim=-1, descending=False)[:, :, 0: self.n_knn].int() # B, M, k
        key_view = grouping_operation(key, idx_knn_view) # B, dim, M, k
        attention_view = self.attn_mlp_view(key_view)  # B, dim, M, k
        attention_view = torch.softmax(attention_view, -1)
        value_view = grouping_operation(value, idx_knn_view) # B, dim, M, k
        agg_view = einsum('b c i j, b c i j -> b c i', attention_view, value_view)  # B, dim, M
        feat_view = torch.bmm(torch.softmax(cos_sim.permute(0, 2, 1), dim=-1), agg_view.permute(0, 2, 1)).permute(0, 2, 1)
        # geo-guided
        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn_geo = query_knn(self.n_knn,
                                pos_flipped,
                                pos_flipped,
                                include_self)
        key_geo = grouping_operation(key, idx_knn_geo)  # B, dim, N, k
        key_geo_ = key.unsqueeze(-1) - key_geo
        pos_ = pos.unsqueeze(-1) - grouping_operation(pos, idx_knn_geo)  # B, 3, N, k
        pos_embedding = self.pos_mlp(pos_)
        attention_geo = self.attn_mlp_geo(key_geo_ + pos_embedding)  # B, dim, N, k
        attention_geo = torch.softmax(attention_geo, -1)
        value_geo = value.unsqueeze(-1) + pos_embedding
        feat_geo = einsum('b c i j, b c i j -> b c i', attention_geo, value_geo)  # B, dim, N
        feat_out = identity + self.conv_view(feat_view) + self.conv_geo(feat_geo)
        return feat_out


class ViewGuidedUpLayer(nn.Module):
    def __init__(self, dim_feat, up_factor=2):
        super().__init__()
        self.dim_ps = 64
        self.mlp_1 = MLP_CONV(in_channel=3,
                              layer_dims=[int(dim_feat / 2), dim_feat])
        self.mlp_2 = MLP_CONV(in_channel=dim_feat * 3,
                              layer_dims=[dim_feat * 2, dim_feat])
        self.vgtransformer = ViewGeoAwareTransformer(in_channel=dim_feat,
                                                   dim=int(dim_feat / 2))
        self.mlp_ps = MLP_CONV(in_channel=dim_feat,
                               layer_dims=[int(dim_feat / 2), self.dim_ps])
        self.ps = nn.ConvTranspose1d(self.dim_ps,
                                     dim_feat,
                                     up_factor,
                                     up_factor,
                                     bias=False)
        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=dim_feat * 2,
                                         hidden_dim=dim_feat,
                                         out_dim=dim_feat)
        self.mlp_delta = MLP_CONV(in_channel=dim_feat,
                                  layer_dims=[int(dim_feat / 2), 3])

    def forward(self, pcd_prev, feat_pc, feat_im, K_prev=None):
        """
        Args:
            pcd_prev: (B, 3, N)
            feat_pc: (B, dim_feat, N)
            feat_im: (B, dim_feat, M)
            K_prev: None or (B, dim_feat, N)
        Returns:
            pcd_child: (B, 3, N*up_factor)
            K_curr: (B, dim_feat, N*up_factor)
        """
        _, _, n = pcd_prev.shape
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, -1, keepdim=True)[0].repeat(1, 1, n),
                            torch.max(feat_pc, -1, keepdim=True)[0].repeat(1, 1, n)], 1)
        Q = self.mlp_2(feat_1)
        H = self.vgtransformer(pcd_prev,
                               feat_im,
                               K_prev if K_prev is not None else feat_pc,
                               Q)
        feat_child = self.mlp_ps(H)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        H_up = self.up_sampler(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))
        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr)))
        pcd_child = self.up_sampler(pcd_prev)
        pcd_child = pcd_child + delta
        return pcd_child, K_curr


if __name__ == '__main__':
    import numpy as np
    pc = torch.randn(16, 256, 256).cuda()
    im = torch.randn(16, 192, 256).cuda()
    model = InterlacedTransformer().cuda()
    out = model(pc, im)
    print(out.shape)