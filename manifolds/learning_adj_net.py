import torch
import torch.nn as nn
import torch.nn.functional as F
from manifolds.att_layers import DenseAtt
# from Norm.norm import Norm, RiemannianGroupNorm  ## added by Cole
from manifolds.ball import PoincareBall as Frechet_Poincare
import math
from manifolds.hyp_layers_v2 import HypLinear, HypAct, community_local_cluster, HyperbolicGraphConvolution

"""
构建方法：1时间序列使用hypformer来构建。    2使用hypformer+特征选择的edge
学习的edge➕threshold约束
"""


class Learning_Adj_Net(nn.Module):
    def __init__(self, in_fea=116, out_fea=116, n_head=1):
        super().__init__()
        self.manifold = Frechet_Poincare()
        self.wq = HypLinear(Frechet_Poincare(), in_fea, out_fea, torch.tensor([1.0]).cuda(), 0.3, True)
        self.wk = HypLinear(Frechet_Poincare(), in_fea, out_fea, torch.tensor([1.0]).cuda(), 0.3, True)
        self.wv = HypLinear(Frechet_Poincare(), in_fea, out_fea, torch.tensor([1.0]).cuda(), 0.3, True)
        self.num_heads = n_head
        self.scale = nn.Parameter(torch.tensor([math.sqrt(out_fea)]))
        # self.wk = nn.ModuleList()
        # self.wq = nn.ModuleList()
        # self.wv = nn.ModuleList()
        # for i in range(self.num_heads):
        #     self.wk.append(HypLinear(Frechet_Poincare(), in_fea, out_fea, torch.tensor([1.0]).cuda(), 0.3, True))
        #     self.wq.append(HypLinear(Frechet_Poincare(), in_fea, out_fea, torch.tensor([1.0]).cuda(), 0.3, True))
        #     self.wv.append(HypLinear(Frechet_Poincare(), in_fea, out_fea, torch.tensor([1.0]).cuda(), 0.3, True))

    def poincare_distance(self, x, y):
        """Compute the Poincaré distance between two points x and y on the unit disk."""
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        norm_y = torch.norm(y, dim=-1, keepdim=True)

        # Compute the distance using the Poincaré distance formula
        dist = torch.acosh(1 + 2 * torch.sum((x - y) ** 2, dim=-1) / ((1 - norm_x ** 2) * (1 - norm_y ** 2)))
        return dist

    def hyperbolic_softmax(self, Q_map):
        """
        对Poincaré圆盘上的Q_map矩阵应用类似softmax的归一化操作。

        参数:
            Q_map (torch.Tensor): 输入张量，形状为(batch, 116, 116)

        返回:
            torch.Tensor: 归一化后的输出，形状与输入相同。
        """
        batch_size, H, _ = Q_map.shape

        # 计算Q_map中每对元素的双曲距离
        distances = torch.zeros(batch_size, H, H)  # 用于存储每个batch中的双曲距离

        for i in range(batch_size):
            for j in range(H):
                for k in range(H):
                    distances[i, j, k] = self.poincare_distance(Q_map[i, j], Q_map[i, k])

        # 使用负的双曲距离计算相似性
        similarities = torch.exp(-distances)  # 计算负距离的指数

        # 对相似性进行归一化（类似于softmax）
        similarity_sum = similarities.sum(dim=-1, keepdim=True)  # 对每个行进行求和
        normalized_similarities = similarities / similarity_sum  # 归一化

        return normalized_similarities

    def forward(self, x):
        # input x : bat, 116, 116
        # q_list = []
        # k_list = []
        # v_list = []
        # for i in range(self.num_heads):
        # batch, 116, out_fea
        x_q = self.wq(x)
        x_k = self.wk(x)
        x_v = self.wv(x)
        Q_mapped = self.manifold.mobius_matvec(x_q, x_k, torch.tensor([1.0]).cuda())
        # att_weight = self.hyperbolic_softmax(Q_mapped)
        att_weight = nn.Softmax(dim=-1)(Q_mapped)
        att_output = self.manifold.mobius_matvec(att_weight, x_v.transpose(-1, -2), torch.tensor([1.0]).cuda())
        # query = torch.stack(q_list, dim=1)  # [bat, H, D, out_fea]
        # key = torch.stack(k_list, dim=1)  # [bat, H, D, out_fea]
        # value = torch.stack(v_list, dim=1)  # [bat, H, D, out_fea]
        return att_output, att_weight


class HyperbolicDomainAdaptLayer(nn.Module):
    def __init__(self, node_num_comm, k=7, out_fea=32, c=torch.tensor([1.0]).cuda(), site_num=4):
        super().__init__()
        self.c = c
        self.k = k
        self.manifold = Frechet_Poincare()
        self.out_fea = out_fea
        self.GCN = HyperbolicGraphConvolution()
        self.comm_layers = nn.ModuleList([])
        for comm_i in range(k):
            self.comm_layers.append(community_local_cluster(node_num=node_num_comm[comm_i]))
        self.conv1 = nn.Conv2d(1, out_fea, (self.k, 64), stride=1)
        self.hyp_linear = HypLinear(Frechet_Poincare(), out_fea, 2, torch.tensor([1.0]).cuda(), 0.3, True)
        # self.domain_class = AdversarialNetwork()
        # self.trans_encoder = Transformer(depth=1, token_num=116, token_length=116, kernal_length=31, dropout=0.3)
        # self.ffn = FFN(channels=116, dropout=0.3)
        self.edge_atten = Learning_Adj_Net()
        act = getattr(F, "relu")
        self.hyp_act = HypAct(self.manifold, c, c, act, norm=None)
        # self.conv_trans_strict = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 116), stride=1, padding=0)
        # self.mlp_trans = nn.Sequential(
        #     nn.Linear(32, 32),
        #     nn.LayerNorm(32),
        #     nn.GELU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(32, 2),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, x, adj, ROIs_belong):
        x_edge = x.clone()
        # x_edge = self.manifold.logmap0(x_edge, self.c)
        A1, atten_weight = self.edge_atten(x_edge)
        # A1 = self.ffn(A1)
        # A = torch.squeeze(A1, dim=1)
        adj1 = self.hyp_act(A1 + A1.transpose(1, 2))
        # adj1 = self.manifold.proj(self.manifold.expmap0(adj1, c=self.c), c=self.c)
        # A1 = torch.unsqueeze(adj1, dim=1)
        # A1 = self.conv_trans_strict(A1).reshape(-1, 32, 116)
        # trans_fea = torch.amax(A1, dim=2)
        # edge_class = self.mlp_trans(trans_fea)

        x_gcn, _ = self.GCN(x, adj1)
        x_t = self.manifold.logmap0(x_gcn, self.c)  # return to Euclidean space
        comm_fea = torch.zeros((32, 1, 64))
        for comm_i in range(self.k):
            if comm_i == 0:
                comm_fea = self.comm_layers[comm_i](x_t[:, ROIs_belong[comm_i]])
            else:
                comm_temp = self.comm_layers[comm_i](x_t[:, ROIs_belong[comm_i]])
                comm_fea = torch.cat((comm_fea, comm_temp), dim=1)
        # comm_fea = self.manifold.proj(self.manifold.expmap0(comm_fea, c=self.c), c=self.c)
        # reverse_bottleneck = ReverseLayerF.apply(comm_fea, 0.1)
        brain_fea = torch.unsqueeze(comm_fea, dim=1)  # bat 1 k 64
        brain_fea_1 = self.conv1(brain_fea).reshape(-1, self.out_fea)
        brain_fea = self.manifold.proj(self.manifold.expmap0(brain_fea_1, c=self.c), c=self.c)
        brain_fea = self.hyp_linear(brain_fea)
        # domain_out = self.domain_class(reverse_bottleneck)
        return brain_fea  # , brain_fea  # , domain_out
