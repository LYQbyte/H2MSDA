import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from torch.autograd import Function
from manifolds.att_layers import DenseAtt
# from Norm.norm import Norm, RiemannianGroupNorm  ## added by Cole
from manifolds.ball import PoincareBall as Frechet_Poincare
from feature_extractor import Transformer
from new_attention import FFN


# from files.frechet_agg import frechet_agg

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return beta * z, beta


class Atten(nn.Module):
    def __init__(self, batch_size=32, N=116, dropout=0.3):
        super(Atten, self).__init__()
        self.node = N
        self.dropout = dropout
        self.batch_size = batch_size
        # self.node_left = left_node_num
        self.cluster_atten = nn.Parameter(torch.ones(batch_size, self.node))
        self.node_atten = nn.Parameter(torch.ones(1, 116, 1))
        self.atten = nn.Sequential(
            nn.Linear(N, N),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        # self.atten_gcn = GCN_layer(32, 1)
        self.atten_gra = nn.Sequential(
            nn.Linear(116, 116),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(116, 116),
            nn.ReLU()
        )
        self.batn = nn.BatchNorm1d(N)

    def forward(self, gcn_out):   # gcn_out: bat 116 32
        # out1 = torch.amax(gcn_out, 2)  # batch * 116
        # out1 = torch.ones((out1.shape[0], self.node)).to(torch.float32).cuda()
        # atten_out = self.atten(out1)
        # # atten_gra = self.atten_gra(gra).reshape(-1, self.node, 1)
        # atten_out = atten_out.reshape(-1, self.node, 1)  # get bat * 116 * 1
        out2 = self.batn(gcn_out * self.node_atten + gcn_out)  # bat * 116 * 32
        return out2


def get_dim_act_curv(args=None, feat_dim=116, dim=16, num_layer=2, device='cuda', c=1.0):
    """
    Helper function to get dimension and activation at every layer.
    :param args:bias
    :return:
    """
    # if not args.act:
    #     # act = lambda x: x
    #     act = None
    # else:
    act = getattr(F, 'selu')
    acts = [act] * (num_layer - 1)
    dims = [feat_dim] + ([dim] * (num_layer - 1))

    use_acts = [True] * (num_layer - 1)
    print(dims, 'dims')
    if args.task in ['lp', 'rec']:
        dims += [dim]
        acts += [act]
        use_acts += [True]

        n_curvatures = num_layer
    else:
        n_curvatures = num_layer - 1
    ### add our stuff
    # if hasattr(args, 'output_dim') and args.output_dim > 0:
    #     dims[-1] = args.output_dim
    # else:
    dims[-1] = dim

    # if hasattr(args,
    #            'output_act'):  ## if has arg, change to last to final function, regardless of whether we added extra dim or not
    #     if args.output_act not in ('None', None):
    #         acts[-1] = getattr(F, args.output_act)
    #         use_acts[-1] = True
    #     else:
    #         acts[-1] = act
    #         use_acts[-1] = False
    # else:
    acts[-1] = act
    use_acts[-1] = True

    print(dims, 'dims after')
    # if args.c is None:
    #     # create list of trainable curvature parameters
    #     curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    # else:
    # fixed curvature
    curvatures = [torch.tensor([c]) for _ in range(n_curvatures)]
    # if not args.cuda == -1:
    curvatures = [curv.to(device) for curv in curvatures]
    return dims, acts, curvatures, use_acts


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias, use_act=True, norm=None):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act, norm=norm)
        self.use_act = use_act

    def forward(self, x):
        h = self.linear.forward(x)
        if self.use_act:  ## careful w/ c = None, bc the activation projects from c in to c out
            h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold=Frechet_Poincare(), in_features=116, out_features=64, c_in=torch.tensor([1.0]).cuda(),
                 c_out=torch.tensor([1.0]).cuda(), dropout=0.3, act='selu', use_bias=1, use_att=0, local_agg=1,
                 use_frechet_mean=0, use_act=True, norm=None, use_agg=True):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        # print*use_bias
        #  ## YOOOOOO implement the hyperbolic aggregation here!!!!!!
        #  ## also... seems ideal for hyperbolic graphnorm!!!!
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg, use_frechet_mean, use_agg=use_agg)
        # self.agg.args = args
        act = getattr(F, act)

        if use_act:
            self.hyp_act = HypAct(manifold, c_in, c_out, act, norm=norm)
        self.use_act = use_act
        # self.most_recent={'in':None,'out':None}
        # self.c_in=c_cin
        # self.
        # self.in_features

    def forward(self, x, adj):
        # x, adj = input
        # print(x.shape,'New layer!')
        # print(x.min(),x.max(),'forward, act=',self.use_act)
        # print(x.dtype,'insides')
        self.most_recent = {}
        self.most_recent['in'] = x
        h = self.linear.forward(x)  ### it's all in the agg?
        h_singles = torch.zeros_like(h)
        if len(x.shape) > 2:
            for i in range(x.shape[0]):  ### size of batch
                h_i = self.linear.forward(x[i])
                h_singles[i] = h_i

            # h = self.agg.forward(h, adj)
            h = h_singles
        h_singles = torch.zeros_like(h)
        if len(x.shape) > 2:
            for i in range(x.shape[0]):  ### size of batch
                # h_i=self.linear.forward(x[i])
                h_i = self.agg.forward(h[i], adj[i])
                h_singles[i] = h_i
                # print(h_i,'single')
                # print(h[i],'fu;;')
                # print(h_i==h[i],'EQUALS')

            h = h_singles
        else:
            h = self.agg.forward(h, adj)
        h_singles = torch.zeros_like(h)

        # print(h.min(),h.max(),'after linear agg=',self.use_act)
        if self.use_act:  ## careful w/ c = None, bc the activation projects from c in to c out
            #
            if len(x.shape) > 2:
                for i in range(x.shape[0]):  ### size of batch
                    # h_i=self.linear.forward(x[i])
                    h_i = self.hyp_act.forward(h[i])
                    h_singles[i] = h_i
                    # print(h_i,'single')
                    # print(h[i],'fu;;')
                    # print(h_i==h[i],'EQUALS')
                h = h_singles
            else:
                h = self.hyp_act.forward(h)

        output = h, adj

        self.most_recent['out'] = h
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        # self.most_recent={'in':None,'out':None}

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        self.most_recent = {}
        self.most_recent['in'] = x
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)

        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)  ### just doe
        res = self.manifold.proj(mv, self.c)  # 实现标准化
        if self.use_bias:
            # print("USE BIAS")
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        self.most_recent['out'] = res
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg, use_frechet_agg, use_agg=True):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        self.use_agg = use_agg
        self.use_frechet_agg = use_frechet_agg
        print('frechet agg', use_frechet_agg)
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)
        if self.use_frechet_agg:
            # print('using again')
            self.frechet_agg_man = Frechet_Poincare()
        # self.most_recent={'in':None,'out':None}

    def forward(self, x, adj):
        self.most_recent = {}
        self.most_recent['in'] = x
        x_tangent = self.manifold.logmap0(x, c=self.c)
        """
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)

        elif self.use_frechet_agg and len(x.shape) <= 2:
            # if batch_calc:
            # if hasattr(self.args, 'frechet_B'):  ### to save a bunch of time
            #     frechet_B = self.args.frechet_B
            # else:
            frechet_B = None
            # assert False,'IN HERE'
            # self.frechet_agg_man = Frechet_Poincare(1/-self.c)
            self.frechet_agg_man = Frechet_Poincare(-self.c)
            # if len(x.shape)>2:

            output = frechet_agg(x=x, adj=adj, man=self.frechet_agg_man, B=frechet_B)
            output = self.manifold.proj(output, c=self.c)
            # print(output.shape)
            # print(output,'output')

            self.most_recent['out'] = output
            return output

        elif self.use_frechet_agg:
            assert False, 'wrong spot'
            self.frechet_agg_man = Frechet_Poincare(1 / -self.c)
            output = torch.zeros_like(x)
            # output_oneper=torch.zeros_like(x)
            # print(x.shape)

            for i in range(x.shape[0]):  ### size of batch
                frech_B = self.args.frech_B_list[i]
                # print(frech_B)
                output_i = frechet_agg(x[i], adj[i], man=self.frechet_agg_man, B=frech_B)
                output_i = self.manifold.proj(output_i, c=self.c)
                output[i] = output_i
                # output_oneper[i]=proj

            return output
            # for frech_B in self.args.frechet_B_ist:


        else:
            """

        # print('nothing')
        support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act, norm=None, hyp_act=False):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act
        # self.hyp_act=hyp_act
        # if not norm:

        self.use_norm = False if not norm else True
        # self.use_norm=False if not norm else True
        self.norm = norm

        # self.most_recent={'in':None,'out':None}
        # self.most_recent['in':x]
        # self.most_recent['out':h]
        # print(self.use_norm)
        # print(self.norm,'NORMAN')
        # else

    # norm_type, self.args.embed_size
    def forward(self, x, use_act=True, use_batch=True, frechet_B=None):  ## if need be, we can make act]]\\\
        # print(x.mean(),x.min(),'input')
        self.most_recent = {}
        self.most_recent['in'] = x

        norm_hyp = False

        if not use_act:
            return x

        # if (self.act == None) and (self.use_norm) and (self.norm.norm_hyp):  #### no need to every log transport!!!
        #     # print('no transports fools')
        #     # kudtddmhdgmhd
        #     output = self.norm(x)
        #     self.most_recent['out'] = output
        #     return output
        # if self.hyp_act:
        #     output=self.act(x)

        #     return output

        xt = self.manifold.logmap0(x, c=self.c_in)
        xt_logmap = xt

        if self.act == None:
            pass
        else:  ### probably should put the act like this
            xt = self.act(xt)

        if self.use_norm and not self.norm.norm_hyp:  ### Norm Causing gradient issues, unclear if directly or indirectly.. further investigation required
            # xt=self.norm(xt,frechet_B=frechet_B)
            # xt=self.norm(xt,frechet_B=frechet_B)
            xt = self.norm(xt)

        # for i in range(len(x)):
        # print(x[i],'X',xt_logmap[i],xt[i],'OUTPUT')

        xt = self.manifold.proj_tan0(xt, c=self.c_out)

        if self.use_norm and self.norm.norm_hyp:  ### Norm Causing gradient issues, unclear if directly or indirectly.. further investigation required
            xt = self.norm(self.manifold.expmap0(xt, c=self.c_out), frechet_B)
            # xt= self.norm()
            proj = self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)
            output = self.norm(proj)
            self.most_recent['out'] = output
            return output  #### what does manifold projection do??

        # print(output,'output!!')

        output = self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)
        self.most_recent['out'] = output
        return output

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class community_local_cluster(nn.Module):
    def __init__(self, node_num, node_length=64, dropout=0.3):
        super(community_local_cluster, self).__init__()
        self.cluster = nn.Sequential(
            nn.Conv1d(node_num, 1, kernel_size=15, padding=15 // 2, stride=1),  #
            nn.LayerNorm(node_length),
            nn.SELU(),  # Tanh  # SELU
            nn.Dropout(dropout),
        )
        # self.trans_encoder = Transformer(depth=1, token_num=node_num, token_length=64, kernal_length=15, dropout=0.3)
        # self.ffn = FFN(channels=64, dropout=0.3)

    def forward(self, x):
        # x = self.trans_encoder(x)
        # x = self.ffn(x)
        x_cluster = self.cluster(x)
        # x_cluster = self.net(x_cluster)
        return x_cluster


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class AdversarialNetwork(nn.Module):
    def __init__(self, k=7, site_num=2, out_fea=32, c=torch.tensor([1.0]).cuda()):
        super().__init__()
        self.out_fea = out_fea
        self.k = k
        self.c = c
        self.manifold = Frechet_Poincare()
        self.conv1 = nn.Conv2d(1, out_fea, (self.k, 32), stride=1)
        self.hyp_linear = HypLinear(Frechet_Poincare(), out_fea, site_num, torch.tensor([1.0]).cuda(), 0.3, True)
        self.euc_linear = nn.Sequential(
            nn.Linear(out_fea, site_num),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x1 = torch.unsqueeze(x, dim=1)
        x2 = self.conv1(x).reshape(-1, self.out_fea)
        brain_fea = self.manifold.proj(self.manifold.expmap0(x2, c=self.c), c=self.c)
        brain_fea = self.hyp_linear(brain_fea)
        return brain_fea


class hyp_attention(nn.Module):
    def __init__(self, in_fea=64, out_fea=64, n_head=1):
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
        # x_v = self.wv(x)
        Q_mapped = self.manifold.mobius_matvec(x_q, x_k, torch.tensor([1.0]).cuda())
        # att_weight = self.hyperbolic_softmax(Q_mapped)
        att_weight = nn.Softmax(dim=-1)(Q_mapped)
        # att_output = self.manifold.mobius_matvec(att_weight, x_v.transpose(-1, -2), torch.tensor([1.0]).cuda())
        # query = torch.stack(q_list, dim=1)  # [bat, H, D, out_fea]
        # key = torch.stack(k_list, dim=1)  # [bat, H, D, out_fea]
        # value = torch.stack(v_list, dim=1)  # [bat, H, D, out_fea]
        return att_weight


class HyperbolicDomainAdaptLayer(nn.Module):
    def __init__(self, node_num_comm, k=7, out_fea=32, c=torch.tensor([1.0]).cuda(), site_num=4):
        super().__init__()
        self.c = c
        self.k = k
        self.manifold = Frechet_Poincare()
        self.out_fea = out_fea
        self.GCN = HyperbolicGraphConvolution()
        self.Comm_GCN = HyperbolicGraphConvolution(in_features=64, out_features=32)
        self.comm_layers = nn.ModuleList([])
        for comm_i in range(k):
            self.comm_layers.append(community_local_cluster(node_num=node_num_comm[comm_i]))
        self.conv1 = nn.Conv2d(1, out_fea, (self.k, 32), stride=1)
        self.hyp_linear = HypLinear(Frechet_Poincare(), out_fea, 2, torch.tensor([1.0]).cuda(), 0.3, True)
        self.domain_class = AdversarialNetwork()
        # self.trans_encoder = Transformer(depth=1, token_num=116, token_length=116, kernal_length=31, dropout=0.3)
        # self.ffn = FFN(channels=116, dropout=0.3)
        # self.atten = Attention(64, hidden_size=64)
        self.atten = Atten()
        self.comm_atten_edge = hyp_attention(in_fea=64, out_fea=64)
        act = getattr(F, "relu")
        self.hyp_act = HypAct(self.manifold, c, c, act, norm=None)
        self.global_adj = nn.Parameter(torch.FloatTensor(k, k), requires_grad=True)
        self.global_adj_node = nn.Parameter(torch.ones(116, 116), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)
        # self.global_adj = nn.Parameter(torch.ones(k, k)/2.0, requires_grad=True)
        # self.conv_trans_strict = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 116), stride=1, padding=0)
        # self.mlp_trans = nn.Sequential(
        #     nn.Linear(32, 32),
        #     nn.LayerNorm(32),
        #     nn.GELU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(32, 2),
        #     nn.Softmax(dim=1)
        # )

    def get_adj(self, x, atten=None, self_loop=True):
        # x: b, node, feature
        adj = self.self_similarity(x)  # b, n, n
        num_nodes = adj.shape[-1]
        if atten is not None:
            adj = F.relu(adj * (atten + atten.transpose(1, 2)))
        else:
            adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
        if self_loop:
            adj = adj + torch.eye(num_nodes).cuda()
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def get_adj_node(self, adj, atten=None, self_loop=True):
        # x: b, node, feature
        num_nodes = adj.shape[-1]
        # if atten is not None:
        #     # adj = F.relu(adj * (atten + atten.transpose(1, 2)))
        #     adj = F.relu(adj+adj.transpose(1, 2))
        # else:
        #     adj = F.relu(adj * (self.global_adj_node + self.global_adj_node.transpose(1, 0)))
        if self_loop:
            adj = adj + torch.eye(num_nodes).cuda()
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def self_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s

    def forward(self, x, adj, ROIs_belong):
        # x_edge = x.clone()
        # x_edge = self.manifold.logmap0(x_edge, self.c)
        # A1, atten_weight = self.edge_atten(x_edge)
        # A1 = self.ffn(A1)
        # A = torch.squeeze(A1, dim=1)
        # adj1 = self.hyp_act(atten_weight + atten_weight.transpose(1, 2))
        # adj1 = self.manifold.proj(self.manifold.expmap0(adj1, c=self.c), c=self.c)
        # A1 = torch.unsqueeze(adj1, dim=1)
        # A1 = self.conv_trans_strict(A1).reshape(-1, 32, 116)
        # trans_fea = torch.amax(A1, dim=2)
        # edge_class = self.mlp_trans(trans_fea)
        # adj = self.get_adj_node(adj)
        x_gcn, _ = self.GCN(x, adj)
        x_t = self.manifold.logmap0(x_gcn, self.c)    # return to Euclidean space
        # x_t = self.atten(x_t)
        comm_fea = torch.zeros((32, 1, 64))
        for comm_i in range(self.k):
            if comm_i == 0:
                comm_fea = self.comm_layers[comm_i](x_t[:, ROIs_belong[comm_i]])
            else:
                comm_temp = self.comm_layers[comm_i](x_t[:, ROIs_belong[comm_i]])
                comm_fea = torch.cat((comm_fea, comm_temp), dim=1)
        # adj_comm = self.get_adj(comm_fea)
        comm_fea = self.manifold.proj(self.manifold.expmap0(comm_fea, c=self.c), c=self.c)
        # atten_comm_edge = self.comm_atten_edge(comm_fea)
        adj_comm = self.get_adj(comm_fea)
        fea_1_mmd = comm_fea.reshape(comm_fea.shape[0], -1)
        # adj_comm = self.manifold.proj(self.manifold.expmap0(adj_comm, c=self.c), c=self.c)
        comm_gcn, _ = self.Comm_GCN(comm_fea, adj_comm)
        # brain_fea = comm_gcn.view(comm_gcn.size()[0], -1)
        fea_mmd = comm_gcn.reshape(comm_gcn.shape[0], -1)
        comm_gcn = self.manifold.logmap0(comm_gcn, self.c)
        brain_fea = torch.unsqueeze(comm_gcn, dim=1)  # bat 1 k 64
        # fea_mmd = brain_fea.reshape(brain_fea.shape[0], -1)
        # reverse_bottleneck = ReverseLayerF.apply(brain_fea, 0.5)
        brain_fea_1 = self.conv1(brain_fea).reshape(-1, self.out_fea)
        brain_fea = self.manifold.proj(self.manifold.expmap0(brain_fea_1, c=self.c), c=self.c)
        output = self.hyp_linear(brain_fea)
        # domain_out = self.domain_class(reverse_bottleneck)
        return fea_1_mmd, fea_mmd, output, brain_fea   # , brain_fea  # , domain_out


if __name__ == '__main__':
    model = HyperbolicGraphConvolution().cuda()
    temp_data = torch.randn((32, 116, 116)).cuda()
    adj = torch.randn((32, 116, 116)).cuda()
    output = model(temp_data, adj)

