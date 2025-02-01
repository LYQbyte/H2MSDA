import torch

from files.ball_backward import frechet_ball_backward
from files.hyperboloid_backward import frechet_hyperboloid_backward
from files.ball_forward import frechet_ball_forward
from files.hyperboloid_forward import frechet_hyperboloid_forward
# from manifolds import Lorentz, Poincare, get_manifold_id
from files.ball import Poincare
from files.hyperboloid_lorentz import Lorentz

from files.utils import TOLEPS


def get_manifold_id(x):
    if isinstance(x, Poincare):
        return 0
    elif isinstance(x, Lorentz):
        return 1
    else:
        raise NotImplementedError

#TODO don't need to include this
def to_ball(x, K):
    R = 1 / (-K).sqrt()
    return R * x[..., 1:] / (R + x[..., :1])

def to_hyperboloid(x, K):
    R = 1/ (-K).sqrt()
    xnormsq = x.norm(dim=-1, keepdim=True).pow(2)
    sec_part = 2 * R.pow(2) * x / (R.pow(2) - xnormsq)
    first_part = R * (R.pow(2) + xnormsq) / (R.pow(2) - xnormsq)
    return torch.cat((first_part, sec_part), dim=-1)


class FrechetMean(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, w, K, manifold_id):
        if manifold_id==0:
            mean = frechet_ball_forward(x, w, K, rtol=TOLEPS[x.dtype], atol=TOLEPS[x.dtype])
        elif manifold_id==1:
            mean = frechet_hyperboloid_forward(x, w, K, rtol=TOLEPS[x.dtype], atol=TOLEPS[x.dtype])
        else:
            raise NotImplementedError

        manifold_id = torch.tensor(manifold_id)
        ctx.save_for_backward(x, mean, w, K, manifold_id)
        return mean

    @staticmethod
    def backward(ctx, grad_output):
        X, mean, w, K, manifold_id = ctx.saved_tensors
        manifold_id = manifold_id.item()

        if manifold_id == 0:
            dx, dw, dK = frechet_ball_backward(X, mean, grad_output, w, K)
        elif manifold_id == 1:
            dx, dw, dK = frechet_hyperboloid_backward(X, mean, grad_output, w, K)
        else:
            raise NotImplementedError
        return dx, dw, dK, None

def frechet_mean(x, manifold, w=None,return_numpy=False):
    if w is None:
        w = torch.ones(x.shape[:-1]).to(x)
    # print(w, manifold.K,'X, W , manifold.K')
    # print(get_manifold_id(manifold),'MANIFOLD!!')
    mean = FrechetMean.apply(x, w, manifold.K, get_manifold_id(manifold))

    if return_numpy:
        mean=mean.numpy()
    return mean
