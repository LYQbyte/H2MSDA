from brainspace.gradient import GradientMaps
import numpy as np
import torch
from scipy.stats import spearmanr
from scipy import sparse as ssp
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian
from scipy.sparse.csgraph import connected_components
import warnings
from sklearn.utils import check_random_state


def spearman_correlation_matrix(x):
    # 假设 x 是 torch.Tensor 格式，转换为 numpy 数组
    x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    # 使用 spearmanr 计算相关矩阵
    corr, _ = spearmanr(x, axis=1)
    return np.array(corr)  # 如果需要返回到 torch.Tensor 格式


def _dominant_set_dense(s, k=int(0.1 * 116), is_thresh=False, norm=False, copy=True):
    """Compute dominant set for a dense matrix."""

    if is_thresh:
        s = s.copy() if copy else s
        s[s <= k] = 0

    else:  # keep top k
        nr, nc = s.shape
        idx = np.argpartition(s, nc - k, axis=1)
        row = np.arange(nr)[:, None]
        if copy:
            col = idx[:, -k:]  # idx largest
            data = s[row, col]
            s = np.zeros_like(s)
            s[row, col] = data
        else:
            col = idx[:, :-k]  # idx smallest
            s[row, col] = 0

    if norm:
        s /= np.nansum(s, axis=1, keepdims=True)

    return s


def is_symmetric(x, tol=1E-10):
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError('Array is not square.')

    if ssp.issparse(x):
        if x.format not in ['csr', 'csc', 'coo']:
            x = x.tocoo(copy=False)
        dif = x - x.T
        return np.all(np.abs(dif.data) < tol)

    return np.allclose(x, x.T, atol=tol)


def make_symmetric(x, check=True, tol=1E-10, copy=True, sparse_format=None):
    if not check or not is_symmetric(x, tol=tol):
        if copy:
            xs = .5 * (x + x.T)
            if ssp.issparse(x):
                if sparse_format is None:
                    sparse_format = x.format
                conversion = 'to' + sparse_format
                return getattr(xs, conversion)(copy=False)
            return xs
        else:
            x += x.T
            if ssp.issparse(x):
                x.data *= .5
            else:
                x *= .5
    return x


def _graph_is_connected(graph):
    return connected_components(graph)[0] == 1


def diffusion_mapping(adj, n_components=10, alpha=0.5, diffusion_time=0,
                      random_state=None):
    """Compute diffusion map of affinity matrix.

    Parameters
    ----------
    adj : ndarray or sparse matrix, shape = (n, n)
        Affinity matrix.
    n_components : int or None, optional
        Number of eigenvectors. If None, selection of `n_components` is based
        on 95% drop-off in eigenvalues. When `n_components` is None,
        the maximum number of eigenvectors is restricted to
        ``n_components <= sqrt(n)``. Default is 10.
    alpha : float, optional
        Anisotropic diffusion parameter, ``0 <= alpha <= 1``. Default is 0.5.
    diffusion_time : int, optional
        Diffusion time or scale. If ``diffusion_time == 0`` use multi-scale
        diffusion maps. Default is 0.
    random_state : int or None, optional
        Random state. Default is None.

    Returns
    -------
    v : ndarray, shape (n, n_components)
        Eigenvectors of the affinity matrix in same order.
    w : ndarray, shape (n_components,)
        Eigenvalues of the affinity matrix in descending order.
    """

    rs = check_random_state(random_state)
    use_sparse = ssp.issparse(adj)

    # Make symmetric
    if not is_symmetric(adj, tol=1E-10):
        warnings.warn('Affinity is not symmetric. Making symmetric.')
        adj = make_symmetric(adj, check=False, copy=True, sparse_format='coo')
    else:  # Copy anyways because we will be working on the matrix
        adj = adj.tocoo(copy=True) if use_sparse else adj.copy()

    # Check connected
    if not _graph_is_connected(adj):
        warnings.warn('Graph is not fully connected.')

    ###########################################################
    # Step 2
    ###########################################################
    # When α=0, you get back the diffusion map based on the random walk-style
    # diffusion operator (and Laplacian Eigenmaps). For α=1, the diffusion
    # operator approximates the Laplace-Beltrami operator and for α=0.5,
    # you get Fokker-Planck diffusion. The anisotropic diffusion
    # parameter: \alpha \in \[0, 1\]
    # W(α) = D^{−1/\alpha} W D^{−1/\alpha}
    if alpha > 0:
        if use_sparse:
            d = np.power(adj.sum(axis=1).A1, -alpha)
            adj.data *= d[adj.row]
            adj.data *= d[adj.col]
        else:
            d = adj.sum(axis=1, keepdims=True)
            d = np.power(d, -alpha)
            adj *= d.T
            adj *= d

    ###########################################################
    # Step 3
    ###########################################################
    # Diffusion operator
    # P(α) = D(α)^{−1}W(α)
    if use_sparse:
        d_alpha = np.power(adj.sum(axis=1).A1, -1)
        adj.data *= d_alpha[adj.row]
    else:
        adj *= np.power(adj.sum(axis=1, keepdims=True), -1)

    ###########################################################
    # Step 4
    ###########################################################
    if n_components is None:
        n_components = max(2, int(np.sqrt(adj.shape[0])))
        auto_n_comp = True
    else:
        auto_n_comp = False

    # For repeatability of results
    v0 = rs.uniform(-1, 1, adj.shape[0])

    # Find largest eigenvalues and eigenvectors
    w, v = eigsh(adj, k=n_components + 1, which='LM', tol=0, v0=v0)

    # Sort descending
    w, v = w[::-1], v[:, ::-1]

    ###########################################################
    # Step 5
    ###########################################################
    # Force first eigenvector to be all ones.
    v /= v[:, [0]]

    # Largest eigenvalue should be equal to one too
    w /= w[0]

    # Discard first (largest) eigenvalue and eigenvector
    w, v = w[1:], v[:, 1:]

    if diffusion_time <= 0:
        # use multi-scale diffusion map, ref [4]
        # considers all scales: t=1,2,3,...
        w /= (1 - w)
    else:
        # Raise eigenvalues to the power of diffusion time
        w **= diffusion_time

    if auto_n_comp:
        # Choose n_comp to coincide with a 95 % drop-off
        # in the eigenvalue multipliers, ref [4]
        lambda_ratio = w / w[0]

        # If all eigenvalues larger than 0.05, select all
        # (i.e., sqrt(adj.shape[0]))
        threshold = max(0.05, lambda_ratio[-1])
        n_components = np.argmin(lambda_ratio > threshold)

        w = w[:n_components]
        v = v[:, :n_components]

    # Rescale eigenvectors with eigenvalues
    v *= w[None, :]

    # Consistent sign (s.t. largest value of element eigenvector is pos)
    v *= np.sign(v[np.abs(v).argmax(axis=0), range(v.shape[1])])
    return v, w


dict_4domain = np.load(r"/SD1/luoyq/ABIDE_site_data/dict_4domain.npy", allow_pickle=True).item()
FC = dict_4domain["fc"]
avg_FC = np.sum(FC, axis=0) / len(FC)

# aff_spear = spearman_correlation_matrix(_dominant_set_dense(avg_FC))
# aff_spear[aff_spear < 0] = 0
# gradient, _ = diffusion_mapping(adj=aff_spear, random_state=1)
# np.save(r"/SD1/luoyq/ABIDE_site_data/site_data_gradient_hyper", gradient)


gm1 = GradientMaps(n_components=10, approach='dm', kernel='spearman', random_state=42)
gm1.fit(avg_FC)
bs_gradient = gm1.gradients_
np.save(r"/SD1/luoyq/ABIDE_site_data/bs_site_gradient_hyper", bs_gradient)
