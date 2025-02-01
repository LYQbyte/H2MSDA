import torch
import torch.nn as nn
import torch.nn.functional as F

# import manifolds_utils
from files.poincare import PoincareBall
from att_layers import GraphAttentionLayer
import hyp_layers as hyp_layers
import math_utils as pmath

from files.norm import Norm, RiemannianGroupNorm ## added by Cole
# from Norm.norm import RiemannianGroupNorm as RiemannianBatchNorm


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output


class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c)
        self.manifold = PoincareBall()   # getattr(manifolds_utils, args.manifold)()
        assert args.num_layers > 1
        # dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        dims, acts, self.curvatures,use_acts = hyp_layers.get_dim_act_curv(args)

        hnn_layers = []
        print(dims,'dims')
        print(acts,'acts')
        print(use_acts,'uise acts')
        use_norm = True if hasattr(args,'use_norm') and args.use_norm>0 else False
        norm_type='bn'
        self.curvatures.append(self.c)
        for i in range(len(dims) - 1): ### implement on here ?
            print(norm_type)
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            norm = Norm(norm_type, out_dim) if ((i<len(dims)-2) and (use_norm)) else None
            # norm = Norm('gn', out_dim) if i<len(dims)-2 else None ## can't be batching output@
            act = acts[i]
            use_act=use_acts[i]
            hnn_layers.append(
                    hyp_layers.HNNLayer(
                            self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias,use_act=use_act,norm=norm,args=args)
            )
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return super(HNN, self).encode(x_hyp, adj)


class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args=None):
        super(HGCN, self).__init__(c)
        self.manifold = PoincareBall()
        # assert args.num_layers > 1
        dims, acts, self.curvatures,use_acts = hyp_layers.get_dim_act_curv(args)
        print(dims,'dims')
        print(acts,'acts')
        print(use_acts,'uise acts')
        self.curvatures.append(self.c)
        hgc_layers = []
        # Norm(norm_type, self.args.embed_size)
        use_frechet_agg = True
        use_output_agg = True
        # use_frechet_agg=False
        use_norm = False
        hyp_act = False

        # if hyp_act:
        #     assert args.act not in ('leaky_relu', 'elu', 'selu')
        # norm_type='bn'
        norm_type='gn'
        # norm_type='rbn'
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            print(norm_type, 'NORMAL TYPE')
            norm = Norm(norm_type, args,out_dim) if ((i<len(dims)-2) and (use_norm)) else None ## can't be batching output@
            use_agg = True if ((i<len(dims)-2) or (use_output_agg)) else False ## can't be batching output@

            print(use_agg,'USE AGG')

            # norm=None
            act = acts[i]

            use_act=use_acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out,
                             0.3, act, 1, 0, 1,use_frechet_agg,use_act=use_act,norm=norm,args=args,use_agg=use_agg
                    )
            )
        # sdd
        # jsjsj
        self.layers_list=hgc_layers
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).encode(x_hyp, adj)


