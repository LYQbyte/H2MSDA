import torch
from manifolds.diff_frech_mean_utils import EPS, cosh, sinh, tanh, arcosh, arsinh, artanh, sinhdiv, divsinh

import abc

import numpy as np
import torch


class Manifold(object):
    """
    Abstract class to define operations on a manifold.
    """

    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2, c):
        """Squared distance between pairs of points."""
        raise NotImplementedError

    def egrad2rgrad(self, p, dp, c):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        raise NotImplementedError

    def proj(self, p, c):
        """Projects point p on the manifold."""
        raise NotImplementedError

    def proj_tan(self, u, p, c):
        """Projects u on the tangent space of p."""
        raise NotImplementedError

    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        raise NotImplementedError

    def expmap(self, u, p, c):
        """Exponential map of u at point p."""
        raise NotImplementedError

    def logmap(self, p1, p2, c):
        """Logarithmic map of point p1 at point p2."""
        raise NotImplementedError

    def expmap0(self, u, c):
        """Exponential map of u at the origin."""
        raise NotImplementedError

    def logmap0(self, p, c):
        """Logarithmic map of point p at the origin."""
        raise NotImplementedError

    def mobius_add(self, x, y, c, dim=-1):
        """Adds points x and y."""
        raise NotImplementedError

    def mobius_matvec(self, m, x, c):
        """Performs hyperboic martrix-vector multiplication."""
        raise NotImplementedError

    def init_weights(self, w, c, irange=1e-5):
        """Initializes random weigths on the manifold."""
        raise NotImplementedError

    def inner(self, p, c, u, v=None, keepdim=False):
        """Inner product for tangent vectors at point x."""
        raise NotImplementedError

    def ptransp(self, x, y, u, c):
        """Parallel transport of u from x to y."""
        raise NotImplementedError

    def ptransp0(self, x, u, c):
        """Parallel transport of u from the origin to y."""
        raise NotImplementedError



# class Poincare(Manifold):
#     def __init__(self, K=-1.0, edge_eps=1e-3):
#         # print(K,'K')
#         # print(self)
#         # print(Manifold)
#         # sadsd
#
#         super(Poincare, self).__init__()
#         self.edge_eps = 1e-3
#         try:
#             assert K < 0
#         except:
#             print(K, 'MUST BE NEGATIVE WHAT THE HELL>>')
#             K -= .1
#             assert K < 0
#         if torch.is_tensor(K):
#             self.K = K
#         else:
#             self.K = torch.tensor(K)
#         # print(self.K,"WHOS K?")
#
#     def sh_to_dim(self, sh):
#         if hasattr(sh, '__iter__'):
#             return sh[-1]
#         else:
#             return sh
#
#     def dim_to_sh(self, dim):
#         if hasattr(dim, '__iter__'):
#             return dim[-1]
#         else:
#             return dim
#
#     def zero(self, *shape):
#         return torch.zeros(*shape)
#
#     def zero_tan(self, *shape):
#         return torch.zeros(*shape)
#
#     def zero_like(self, x):
#         return torch.zeros_like(x)
#
#     def zero_tan_like(self, x):
#         return torch.zeros_like(x)
#
#     def lambda_x(self, x, keepdim=False):
#         return 2 / (1 + self.K * x.pow(2).sum(dim=-1, keepdim=keepdim)).clamp_min(min=EPS[x.dtype])
#
#     def inner(self, x, u, v, keepdim=False):
#         return self.lambda_x(x, keepdim=True).pow(2) * (u * v).sum(dim=-1, keepdim=keepdim)
#
#     def proju(self, x, u):
#         return u
#
#     def projx(self, x):
#         norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS[x.dtype])
#         maxnorm = (1 - self.edge_eps) / (-self.K).sqrt()
#         cond = norm > maxnorm
#         projected = x / norm * maxnorm
#         return torch.where(cond, projected, x)
#
#     def egrad2rgrad(self, x, u):
#         return u / self.lambda_x(x, keepdim=True).pow(2)
#
#     def mobius_addition(self, x, y):
#         x2 = x.pow(2).sum(dim=-1, keepdim=True)
#         y2 = y.pow(2).sum(dim=-1, keepdim=True)
#         xy = (x * y).sum(dim=-1, keepdim=True)
#         num = (1 - 2 * self.K * xy - self.K * y2) * x + (1 + self.K * x2) * y
#         denom = 1 - 2 * self.K * xy + (self.K.pow(2)) * x2 * y2
#         return num / denom.clamp_min(EPS[x.dtype])
#
#     def exp(self, x, u):
#         u_norm = u.norm(dim=-1, keepdim=True).clamp_min(min=EPS[x.dtype])
#         second_term = (
#                 tanh((-self.K).sqrt() / 2 * self.lambda_x(x, keepdim=True) * u_norm) * u / ((-self.K).sqrt() * u_norm)
#         )
#         gamma_1 = self.mobius_addition(x, second_term)
#         return gamma_1
#
#     def log(self, x, y):
#         sub = self.mobius_addition(-x, y)
#         sub_norm = sub.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])
#         lam = self.lambda_x(x, keepdim=True)
#         return 2 / ((-self.K).sqrt() * lam) * artanh((-self.K).sqrt() * sub_norm) * sub / sub_norm
#
#     def dist(self, x, y, squared=False, keepdim=False):
#         dist = 2 * artanh((-self.K).sqrt() * self.mobius_addition(-x, y).norm(dim=-1)) / (-self.K).sqrt()
#         return dist.pow(2) if squared else dist
#
#     def _gyration(self, u, v, w):
#         u2 = u.pow(2).sum(dim=-1, keepdim=True)
#         v2 = v.pow(2).sum(dim=-1, keepdim=True)
#         uv = (u * v).sum(dim=-1, keepdim=True)
#         uw = (u * w).sum(dim=-1, keepdim=True)
#         vw = (v * w).sum(dim=-1, keepdim=True)
#         a = - self.K.pow(2) * uw * v2 - self.K * vw + 2 * self.K.pow(2) * uv * vw
#         b = - self.K.pow(2) * vw * u2 + self.K * uw
#         d = 1 - 2 * self.K * uv + self.K.pow(2) * u2 * v2
#         return w + 2 * (a * u + b * v) / d.clamp_min(EPS[u.dtype])
#
#     def transp(self, x, y, u):
#         return (
#                 self._gyration(y, -x, u)
#                 * self.lambda_x(x, keepdim=True)
#                 / self.lambda_x(y, keepdim=True)
#         )
#
#     def __str__(self):
#         return 'Poincare Ball'
#
#     def squeeze_tangent(self, x):
#         return x
#
#     def unsqueeze_tangent(self, x):
#         return x


class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-12
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def sqdist(self, p1, p2, c):
        ### BATCH

        ## here would be a good spot to normalize?-- no it wouldn't leave this shit alone.
        # print(p1,'p1?')
        # print(p2,'p1?')
        # print(c,'c?')
        # p1=torch.from_numpy(np.array(p1).astype(float))
        # p2=torch.from_numpy(np.array(p2).astype(float))
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )

        # print(dist_c,'DIST C')
        # print(sqrt_c,'sqrizzle')
        dist = dist_c * 2 / sqrt_c
        # print(dist)
        return dist ** 2

    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x, c):
        ##BATCH
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        # print(maxnorm,'MAX NORM',c)
        cond = norm > maxnorm
        projected = x / norm * maxnorm

        return torch.where(cond, projected, x)

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        ##BATCH
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        ## BATCH
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        ### BATCH
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        ### BATCH
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2

        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x, c):
        ### BATCH
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        if len(m.shape)<3:
            mx = x @ m.transpose(-1, -2)
        else:
            mx = torch.bmm(x, m.transpose(-1, -2))
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    def init_weights(self, w, c, irange=1e-5):
        ## BATCH
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        print(d)
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_hyperboloid(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)
