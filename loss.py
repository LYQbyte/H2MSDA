import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.autograd.function import Function
from numpy import random


def rbf_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
        source: 源域数据（n * len(x))
        target: 目标域数据（m * len(y))
        kernel_mul:
        kernel_num: 取不同高斯核的数量
        fix_sigma: 不同高斯核的sigma值
    Return:
        sum(kernel_val): 多个核矩阵之和
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0)
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算源域数据和目标域数据的MMD距离
    Params:
        source: 源域数据（n * len(x))
        target: 目标域数据（m * len(y))
        kernel_mul:
        kernel_num: 取不同高斯核的数量
        fix_sigma: 不同高斯核的sigma值
    Return:
        loss: MMD loss
    """
    n = int(source.size()[0])
    m = int(target.size()[0])
    # batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = rbf_kernel(source, target,
                         kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # # 根据式（3）将核矩阵分成4部分
    # XX = kernels[:batch_size, :batch_size]
    # YY = kernels[batch_size:, batch_size:]
    # XY = kernels[:batch_size, batch_size:]
    # YX = kernels[batch_size:, :batch_size]
    # loss = torch.mean(XX + YY - XY - YX)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum()
    return loss


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def mcc_loss(out_target, temperature=2.5, class_num=2):
    train_bs = out_target.shape[0]   # batch size: 32
    outputs_target_temp = out_target / temperature
    target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
    target_entropy_weight = Entropy(target_softmax_out_temp).detach()
    target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
    target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)
    cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(
        target_softmax_out_temp)
    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
    return mcc_loss


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classified examples (p > .5),
                                   putting more focus on hard, misclassified examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLoss1(nn.Module):
    def __init__(self, alpha=0.25, gamma=1, reduction='mean'):
        """
        alpha: 平衡因子，用来控制不同类别对损失的贡献
        gamma: 调节因子，增加对难分类样本的关注
        reduction: 损失的返回形式 ('mean' 或 'sum')
        """
        super(FocalLoss1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # 计算预测概率
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def mobius_add(x, y, c=torch.tensor([1.0]).cuda()):
    """
    实现莫比乌斯加法
    参数：
        x, y: torch.Tensor, 点的坐标，形状为 (batch_size, dim)
        c: float, 曲率
    返回：
        torch.Tensor, 莫比乌斯加法的结果
    """
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2

    return num / denom.clamp_min(min=1e-9)


def hyperbolic_distance(p1, p2, c=torch.tensor([1.0]).cuda()):
    """
    计算庞加莱圆盘上两点之间的双曲距离
    参数：
        p1, p2: torch.Tensor, 两个点的坐标，形状为 (batch_size, dim)
        c: float, 双曲空间的曲率
    返回：
        torch.Tensor, 双曲距离
    """
    sqrt_c = torch.sqrt(c)  # 曲率的平方根
    mobius_diff = mobius_add(-p1, p2, c)  # 计算莫比乌斯加法的结果
    norm = torch.norm(mobius_diff, dim=-1)  # 计算欧几里得范数
    dist = 2 / sqrt_c * torch.atanh(sqrt_c * norm.clamp(max=1 - 1e-5))  # 防止溢出
    return dist


def tanh(x):
    return x.tanh()
EPS = {torch.float32: 1e-4, torch.float64: 1e-8}

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + EPS[x.dtype], 1 - EPS[x.dtype])
        ctx.save_for_backward(x)
        res = (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)

artanh = Artanh.apply

def mobius_matvec(m, x, c, min_norm=1e-9):
    ### BATCH
    sqrt_c = c ** 0.5
    x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(min_norm)
    if len(m.shape)<3:
        mx = x @ m.transpose(-1, -2)
    else:
        mx = torch.bmm(x, m.transpose(-1, -2))
    mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(min_norm)
    res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
    cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return res


def poincare_distance_matrix(source, target, c=1.0, min_norm=1e-10):
    """
    计算源域和目标域之间的双曲距离矩阵
    source: 源域数据 (n_samples, feature_dim)
    target: 目标域数据 (m_samples, feature_dim)
    c: 曲率
    """
    n_samples = source.size(0)
    m_samples = target.size(0)

    # 扩展 source 和 target 数据，便于批量计算
    source_expanded = source.unsqueeze(1).expand(n_samples, m_samples, source.size(1))
    target_expanded = target.unsqueeze(0).expand(n_samples, m_samples, target.size(1))

    # 使用mobius_matvec计算双曲距离
    dist_matrix = mobius_matvec(target_expanded, source_expanded, c)
    return dist_matrix


def rbf_kernel_matrix(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, c=1.0, min_norm=1e-10):
    """
    计算源域和目标域数据的高斯核矩阵
    source: 源域数据 (n_samples, feature_dim)
    target: 目标域数据 (m_samples, feature_dim)
    kernel_mul: 用于计算带宽的倍数
    kernel_num: 高斯核的数量
    fix_sigma: 固定带宽
    c: 庞加莱圆盘的曲率
    """
    dist_matrix = poincare_distance_matrix(source, target, c, min_norm)
    n_samples = dist_matrix.size(0)
    m_samples = dist_matrix.size(1)

    # 调整高斯核的带宽
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(dist_matrix) / (n_samples * m_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 计算每个高斯核
    kernel_vals = [torch.exp(-dist_matrix / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_vals)  # 返回所有核的和


def hyperbolic_gaussian_kernel(source, target, c=torch.tensor([1.0]).cuda(), gamma=1.0):
    """
    使用双曲距离构建高斯核矩阵
    参数：
        source: torch.Tensor, 源域样本 (n, dim)
        target: torch.Tensor, 目标域样本 (m, dim)
        c: float, 庞加莱圆盘的曲率
        gamma: float, 高斯核的超参数
    返回：
        torch.Tensor, 核矩阵 (n+m, n+m)
    """
    # 合并源域和目标域
    total = torch.cat([source, target], dim=0)  # (n + m, dim)
    n_samples = total.size(0)

    # 计算每对点的双曲距离
    distances = torch.zeros(n_samples, n_samples, device=total.device)
    for i in range(n_samples):
        for j in range(n_samples):
            distances[i, j] = hyperbolic_distance(total[i].unsqueeze(0), total[j].unsqueeze(0), c)
    # 应用高斯核
    kernel_matrix = torch.exp(-gamma * distances ** 2)
    return kernel_matrix


def mmd_poincare(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, c=1.0, min_norm=1e-9):
    n = int(source.size()[0])
    m = int(target.size()[0])
    # batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernel_matrix = rbf_kernel_matrix(source, target, kernel_mul, kernel_num, fix_sigma, c, min_norm)
    # # 根据式（3）将核矩阵分成4部分
    # XX = kernels[:batch_size, :batch_size]
    # YY = kernels[batch_size:, batch_size:]
    # XY = kernels[:batch_size, batch_size:]
    # YX = kernels[batch_size:, :batch_size]
    # loss = torch.mean(XX + YY - XY - YX)
    XX = kernel_matrix[:n, :n]
    YY = kernel_matrix[n:, n:]
    XY = kernel_matrix[:n, n:]
    YX = kernel_matrix[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum()
    return loss


def CORAL(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).cuda() @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).cuda() @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss


def mobius_add_np(x, y, c=1.0):
    """
    实现莫比乌斯加法
    参数：
        x, y: numpy.ndarray, 点的坐标，形状为 (batch_size, dim)
        c: float, 曲率
    返回：
        numpy.ndarray, 莫比乌斯加法的结果
    """
    x2 = np.sum(x ** 2, axis=-1, keepdims=True)
    y2 = np.sum(y ** 2, axis=-1, keepdims=True)
    xy = np.sum(x * y, axis=-1, keepdims=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2

    return num / np.clip(denom, a_min=1e-9, a_max=None)


def hyperbolic_distance_np(p1, p2, c=1.0):
    """
    计算庞加莱圆盘上两点之间的双曲距离
    参数：
        p1, p2: numpy.ndarray, 两个点的坐标，形状为 (batch_size, dim)
        c: float, 双曲空间的曲率
    返回：
        numpy.ndarray, 双曲距离
    """
    sqrt_c = np.sqrt(c)  # 曲率的平方根
    mobius_diff = mobius_add_np(-p1, p2, c)  # 计算莫比乌斯加法的结果
    norm = np.linalg.norm(mobius_diff, axis=-1)  # 计算欧几里得范数
    dist = 2 / sqrt_c * np.arctanh(np.clip(sqrt_c * norm, a_min=None, a_max=1 - 1e-5))  # 防止溢出
    return dist


def compute_category_prototype(points, c=1.0):
    """

    参数:
        points: numpy.ndarray, 点集合 (N, dim)，每一行是一个点。
        c: float, 曲率参数。

    返回:
        numpy.ndarray, 类别原型点 (dim,)
    """
    base_point = points[0]  # 选第一个点作为基准点

    # 对每个点计算相对基准点的移动
    shifted_points = np.array([mobius_add_np(-base_point, p, c) for p in points])

    # 计算欧几里得平均值
    euclidean_mean = np.mean(shifted_points, axis=0)

    # 将平均值映射回原坐标系，得到类别原型
    prototype = mobius_add_np(base_point, euclidean_mean, c)

    return prototype


def mobius_add_tensor(x, y, c=1.0):
    """
    实现莫比乌斯加法（适用于 PyTorch Tensor）

    参数：
        x, y: torch.Tensor, 点的坐标，形状为 (batch_size, dim)
        c: float, 曲率参数

    返回：
        torch.Tensor, 莫比乌斯加法的结果，形状为 (batch_size, dim)
    """
    # 计算各项平方和内积
    x2 = torch.sum(x ** 2, dim=-1, keepdim=True)  # (batch_size, 1)
    y2 = torch.sum(y ** 2, dim=-1, keepdim=True)  # (batch_size, 1)
    xy = torch.sum(x * y, dim=-1, keepdim=True)  # (batch_size, 1)

    # 计算分子和分母
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2

    # 防止除以零，使用 torch.clamp 来确保分母不小于 1e-9
    denom = torch.clamp(denom, min=1e-9)

    return num / denom


def poincare_distance_proto(x, y, eps=1e-5):
    # 确保点在圆盘内
    norm_x = torch.clamp(torch.norm(x, dim=-1, p=2), max=1 - eps)
    norm_y = torch.clamp(torch.norm(y, dim=-1, p=2), max=1 - eps)
    diff = torch.norm(x - y, dim=-1, p=2)

    # 超曲率距离公式
    numerator = 2 * diff ** 2
    denominator = (1 - norm_x ** 2) * (1 - norm_y ** 2)
    dist = torch.arccosh(1 + numerator / (denominator + eps))
    return dist


def compute_category_prototypes_tensor(features, labels, n_classes=2, c=1.0):
    """
    计算每个类别的原型点
    参数:
        features: torch.Tensor, 输入特征向量 (batch_size, feature_dim)。
        labels: torch.Tensor, 类别标签 (batch_size,)。
        n_classes: int, 总类别数。
        c: float, 曲率参数。
    返回:
        torch.Tensor, 每个类别的原型点 (n_classes, feature_dim)。
    """
    device = features.device
    feature_dim = features.shape[1]
    prototypes = torch.zeros((n_classes, feature_dim), device=device)

    for c_idx in range(n_classes):
        # 获取属于当前类别的特征
        class_features = features[labels == c_idx]
        if class_features.shape[0] == 0:
            # return None
            continue  # 如果该类别没有样本，则跳过

        # 选择第一个点作为基准点
        base_point = class_features[0]

        # 对每个点计算相对基准点的移动
        shifted_points = torch.stack([mobius_add_tensor(-base_point, p, c) for p in class_features])

        # 计算欧几里得平均值
        euclidean_mean = torch.mean(shifted_points, dim=0)

        # 将平均值映射回原坐标系，得到类别原型
        prototype = mobius_add_tensor(base_point, euclidean_mean, c)

        prototypes[c_idx] = prototype

    return prototypes


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        random.seed(36972)
        centers = random.randn(num_classes, feat_dim)
        self.centers = nn.Parameter(torch.from_numpy(centers))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None


def mobius_add_pytorch(x, y, c=1.0):
    """
    Compute Möbius addition of two points in the Poincaré ball model.
    Parameters:
        x, y: torch.Tensor, input points with shape (batch_size, dim).
        c: float, curvature.
    Returns:
        torch.Tensor: result of Möbius addition.
    """
    x2 = torch.sum(x**2, dim=-1, keepdim=True)
    y2 = torch.sum(y**2, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    cxy = 2 * c * xy
    denom = 1 + c * x2 * y2 + cxy + 1e-8
    return ((1 + c * y2) * x + (1 - c * x2) * y) / denom


def hyperbolic_distance_pytorch(p1, p2, c=1.0):
    """
    Compute the hyperbolic distance between two points in the Poincaré ball model.
    Parameters:
        p1, p2: torch.Tensor, input points with shape (batch_size, dim).
        c: float, curvature.
    Returns:
        torch.Tensor: hyperbolic distance.
    """
    sqrt_c = torch.sqrt(torch.tensor(c))
    mobius_diff = mobius_add_pytorch(-p1, p2, c)
    norm = torch.linalg.norm(mobius_diff, dim=-1, keepdim=False)
    dist = 2 / sqrt_c * torch.atanh(torch.clamp(sqrt_c * norm, min=1e-8, max=1 - 1e-5))
    return dist


class HyperbolicCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, curvature=1.0, size_average=True):
        super(HyperbolicCenterLoss, self).__init__()
        self.curvature = curvature
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)

        centers_batch = self.centers.index_select(0, label.long())
        dists = hyperbolic_distance_pytorch(feat, centers_batch, c=self.curvature)
        loss = torch.sum(dists) / (batch_size if self.size_average else 1)
        return loss


def hyperbolic_prototype_loss(source_prototypes, target_prototypes, n_classes=2):
    """
    超曲率原型对齐损失。
    Args:
        n_classes: 总类别数。

    Returns:
        loss: 超曲率原型对齐损失标量。
    """
    # 计算原型对齐损失
    loss = 0.0
    for c in range(n_classes):
        if torch.equal(source_prototypes[c], torch.zeros(32).cuda()) or torch.equal(target_prototypes[c], torch.zeros(32).cuda()):
            continue
        loss += hyperbolic_distance_pytorch(source_prototypes[c], target_prototypes[c])
    return loss

