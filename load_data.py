import pandas as pd
import numpy as np
import torch
from poincare import PoincareBall
from torch.utils.data import Dataset, DataLoader


ball = PoincareBall()


def fscore_core(np,nn,xb,xbp,xbn,xkp,xkn):
    '''
    np: number of positive features   Count_ASD
    nn: number of negative features   Count_CN
    xb: list of the average of each feature of the whole instances
    xbp: list of the average of each feature of the positive instances
    xbn: list of the average of each feature of the negative instances
    xkp: list of each feature which is a list of each positive instance
    xkn: list of each feature which is a list of each negatgive instance
    reference: http://link.springer.com/chapter/10.1007/978-3-540-35488-8_13
    '''

    def sigmap (i,np,xbp,xkp):
        return sum([(xkp[k][i]-xbp[i])**2 for k in range(np)])

    def sigman (i,nn,xbn,xkn):
        return sum([(xkn[k][i]-xbn[i])**2 for k in range(nn)])

    n_feature = len(xb)
    fscores = []
    for i in range(n_feature):
        fscore_numerator = (xbp[i]-xb[i])**2 + (xbn[i]-xb[i])**2
        fscore_denominator = (1/float(np-1))*(sigmap(i,np,xbp,xkp))+ \
                             (1/float(nn-1))*(sigman(i,nn,xbn,xkn))
        fscores.append(fscore_numerator/fscore_denominator)

    return fscores


def F_score_mask(FC_data, labels):
    index_ASD, index_CN = np.where(labels == 1), np.where(labels == 0)
    index_ASD, index_CN = np.array(index_ASD)[0], np.array(index_CN)[0]
    Count_ASD = index_ASD.shape[0]
    Count_CN = index_CN.shape[0]
    FC_vec = []
    for i in range(len(FC_data)):
        vec_i = []
        for j in range(116):
            for k in range(j + 1, 116):
                vec_i.append(FC_data[i][j][k])
        FC_vec.append(vec_i)
    FC_vec = np.array(FC_vec)
    ASD_vec = FC_vec[index_ASD]  # xkp
    CN_vec = FC_vec[index_CN]  # xkn

    FC_vec_avg = FC_vec.sum(axis=0) / len(FC_vec)  # xb
    ASD_vec_avg = ASD_vec.sum(axis=0) / len(ASD_vec)  # xbp
    CN_vec_avg = CN_vec.sum(axis=0) / len(CN_vec)  # xbn
    F_score = fscore_core(Count_ASD, Count_CN, FC_vec_avg, ASD_vec_avg, CN_vec_avg, ASD_vec, CN_vec)
    F_score = np.array(F_score)
    save_index = F_score.argsort()[-670:]  # -670
    zero_index = F_score.argsort()[:6000]  # 6000
    filter_vec = np.zeros(6670)
    filter_vec[save_index] = 1.0
    mask = np.zeros((116, 116))
    Count = 0
    for j in range(116):
        for k in range(j + 1, 116):
            mask[j][k] = filter_vec[Count]
            mask[k][j] = filter_vec[Count]
            Count = Count + 1

    row, col = np.diag_indices_from(mask)
    mask[row, col] = 1.0
    return mask


def standardize(x, detrend=False, standardize=True):
    # 如果需要去趋势，可以先去除线性趋势
    if detrend:
        x -= torch.linspace(0, 1, x.size(0)).unsqueeze(1) * (x[-1] - x[0])  # 简单的线性去趋势
    epsilon = 1e-8
    # 标准化：零均值和单位方差
    if standardize:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        x = (x - mean) / (std + epsilon)

    return x


def hyperbolic_cosine_similarity(x, y, c, manifold):
    x1 = torch.norm(x, p=2)
    y1 = torch.norm(y, p=2)
    dot_product = torch.sum(x * y)
    # 计算双曲余弦相似度
    similarity = dot_product / (x1 * y1)

    return similarity


def hyperbolic_cosine_similarity_matrix(X, c, manifold):
    n_regions = X.shape[0]  # 脑区数目，116
    similarity_matrix = torch.zeros(n_regions, n_regions)  # 初始化116x116的相关矩阵

    # 计算每两个脑区之间的余弦相似度
    for i in range(n_regions):
        for j in range(i, n_regions):  # 上三角矩阵对称性
            # 计算两个脑区之间的双曲余弦相似度
            similarity = hyperbolic_cosine_similarity(X[i], X[j], c, manifold)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # 对称性

    return similarity_matrix


from scipy.stats import spearmanr


def spearman_correlation_matrix(x):
    # 假设 x 是 torch.Tensor 格式，转换为 numpy 数组
    x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    # 使用 spearmanr 计算相关矩阵
    corr, _ = spearmanr(x, axis=1)
    return np.array(corr)  # 如果需要返回到 torch.Tensor 格式


def hyperbolic_FC_cal(data):
    FC_hyperbolic = []
    for i in range(len(data)):
        temp = torch.tensor(data[i]).transpose(0, 1)
        temp = standardize(temp)
        temp_hyp = ball.expmap0(temp, 1.0)
        temp_hyp = ball.proj(temp_hyp, c=1.0)
        FC_i = spearman_correlation_matrix(temp_hyp)
        # FC_i = spearman_correlation_matrix(temp_hyp)
        FC_hyperbolic.append(np.array(FC_i))

    FC_hyperbolic = np.array(FC_hyperbolic)
    return FC_hyperbolic


def load_data_site(site_name="NYU"):
    # shape: num, time_point, rois_num
    site_data = np.array(data[site_name])
    # "DX_GROUP", "AGE_AT_SCAN", "SEX","FIQ","VIQ", "PIQ"
    # 其中1是ASD, 2是CN
    site_info = np.array(data_info[site_name])
    site_label = site_info[:, 0]
    site_info = site_info[:, 1:]

    # shuffle
    indices1 = np.random.permutation(site_data.shape[0])
    site_data = site_data[indices1]
    site_info = site_info[indices1]
    site_label = site_label[indices1]
    site_fc = hyperbolic_FC_cal(site_data)

    # site_info中的-9999用平均值替换
    site_info[:, -3:][site_info[:, -3:] == -9999] = np.nan
    col_means = np.nanmean(site_info[:, -3:], axis=0)
    inds = np.where(np.isnan(site_info[:, -3:]))  # 找到 NaN 的位置
    site_info[:, -3:][inds] = np.take(col_means, inds[1])

    site_label[site_label == 2] = 0
    return site_fc, site_info, site_label


def site_mix(fold=5, site_names=['NYU', 'UM', 'USM', 'UCLA']):
    total_data = []
    labels = []
    len_site = []
    total_info = []
    site_labels = []
    dict_site = {}
    dict_info = {}
    dict_label = {}
    for i in range(len(site_names)):
        site_data, site_info, site_label = load_data_site(site_names[i])
        dict_site[site_names[i]] = site_data
        dict_info[site_names[i]] = site_info
        dict_label[site_names[i]] = site_label
        len_site.append(len(site_label) // fold)

    for i in range(fold - 1):
        for k in range(len(site_names)):
            total_data.append(dict_site[site_names[k]][i * len_site[k]:(i + 1) * len_site[k]])
            total_info.append(dict_info[site_names[k]][i * len_site[k]:(i + 1) * len_site[k]])
            labels.append(dict_label[site_names[k]][i * len_site[k]:(i + 1) * len_site[k]])
            for j in range(len_site[k]):
                site_labels.append(k)

    for k in range(len(site_names)):
        total_data.append(dict_site[site_names[k]][(fold-1) * len_site[k]:])
        total_info.append(dict_info[site_names[k]][(fold-1) * len_site[k]:])
        labels.append(dict_label[site_names[k]][(fold-1) * len_site[k]:])
        for j in range(len(dict_label[site_names[k]][(fold-1) * len_site[k]:])):
            site_labels.append(k)

    total_data = np.concatenate(np.array(total_data), axis=0)
    total_info = np.concatenate(np.array(total_info), axis=0)
    labels = np.concatenate(np.array(labels), axis=0)
    site_labels = np.array(site_labels)
    return total_data, total_info, labels, site_labels


def all_data():
    total_hyperFc = []
    total_labels = []
    total_info = []
    # ts = np.load(r"D:\EEGLab\Data\data_down\dparsf_aal\site_dict_ts.npy", allow_pickle=True).item()
    # site_info = np.load(r"D:\EEGLab\Data\data_down\dparsf_aal\site_info.npy", allow_pickle=True).item()
    site = ['Caltech', 'CMU', 'KKI', 'Leuven', 'MaxMun', 'NYU', 'OHSU', 'Olin', 'Pitt', 'SBL', 'SDSU', 'Stanford',
            'Trinity', 'UCLA', 'UM', 'USM', 'Yale']
    for s in site:
        ts_s = np.array(data[s])
        site_info = np.array(data_info[s])
        site_label = site_info[:, 0]
        site_info = site_info[:, 1:]
        site_info[:, -3:][site_info[:, -3:] == -9999] = np.nan
        col_means = np.nanmean(site_info[:, -3:], axis=0)
        inds = np.where(np.isnan(site_info[:, -3:]))  # 找到 NaN 的位置
        site_info[:, -3:][inds] = np.take(col_means, inds[1])
        site_label[site_label == 2] = 0
        site_fc = hyperbolic_FC_cal(ts_s)
        total_hyperFc.append(site_fc)
        total_labels.append(site_label)
        total_info.append(site_info)
    total_hyperFc = np.concatenate(np.array(total_hyperFc), axis=0)
    total_labels = np.concatenate(np.array(total_labels), axis=0)
    total_info = np.concatenate(np.array(total_info), axis=0)
    return total_hyperFc, total_labels, total_info

data = np.load("/SD1/luoyq/ABIDE_site_data/site_dict_ts.npy", allow_pickle=True).item()
# data = np.load("/SD1/luoyq/MDD/dict_mdd_domain.npy", allow_pickle=True).item()
data_info = np.load("/SD1/luoyq/ABIDE_site_data/site_info.npy", allow_pickle=True).item()
if __name__ == "__main__":
    # loading 4 domain data
    dict_4domain = {}
    total_data, total_info, labels, site_labels = site_mix()
    dict_4domain['fc'] = total_data
    dict_4domain['info'] = total_info
    dict_4domain['label'] = labels
    dict_4domain['site_labels'] = site_labels
    np.save("/SD1/luoyq/ABIDE_site_data/dict_4domain.npy", dict_4domain)
    print("----------FINISHED----------")

    # # loading all sites data
    # site_names = ['Caltech', 'CMU', 'KKI', 'Leuven', 'MaxMun', 'NYU', 'OHSU', 'Olin', 'Pitt', 'SBL', 'SDSU', 'Stanford',
    #         'Trinity', 'UCLA', 'UM', 'USM', 'Yale']
    # total_hyperFc, total_info, total_labels, site_labels = site_mix(fold=10, site_names=site_names)
    # dict_4all = {}
    # dict_4all['fc'] = total_hyperFc
    # dict_4all['info'] = total_info
    # dict_4all['label'] = total_labels
    # np.save("/SD1/luoyq/ABIDE_site_data/dict_4all.npy", dict_4all)

    # loading MDD domain data
    # 选取S20-533、s21-156、s1-148、S25-152、S8-150
    # total_data, total_info, labels, site_labels = site_mix(site_names=["S20", "S21", "S1", "S25", "S8"])

