import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import time
import deepdish as dd
import torch.distributions as tdist
from load_data import F_score_mask
import os
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random
from loss import *
import geoopt
from itertools import cycle
# import copy
# from network import AdversarialNetworkSp
# from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
# from pyclustering.utils.metric import distance_metric, type_metric
# from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer
# from feature_extractor import feature_extractor
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from brainspace.gradient import GradientMaps
from sklearn.cluster import KMeans
from tqdm import tqdm
from hemisphere_net_v2 import Hemi_attention
from utils import hyperbolic_dist_kernel, expmap0
from manifolds.hyp_layers_v3 import HyperbolicDomainAdaptLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-15
site = ['NYU', 'UM', 'USM', 'UCLA']


def confusion(g_turth, predictions):
    tn, fp, fn, tp = confusion_matrix(g_turth, predictions).ravel()
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    sensitivity = (tp) / (tp + fn)
    specificty = (tn) / (tn + fp)
    return accuracy, sensitivity, specificty


def gradient_avg_mask(Fc_all, k, mask=False, node_num=116):
    # Fc_all shape: num 116 116
    # 聚类为k个簇
    # avg_Fc = np.sum(Fc_all, axis=0)/len(Fc_all)
    # gm1 = GradientMaps(n_components=2,  approach='dm', kernel='cosine', random_state=42)
    # gm1.fit(avg_Fc)
    # gradient = gm1.gradients_
    gradient = np.load("/SD1/luoyq/shuffle_good/gradient_hyper_spear.npy")[:, 0:2]   #   /SD1/luoyq/ABIDE_site_data/bs_site_gradient_hyper.npy
    y_pred = KMeans(n_clusters=k, init="k-means++", random_state=1).fit_predict(gradient)
    # y_pred = consensus_clustering(gradient, n_clusters=k, n_kmeans=5)
    # gmm = GMM(n_components=k, random_state=1).fit(gradient)  # 指定聚类中心个数为4
    # y_pred = gmm.predict(gradient)
    if mask == False:
        cluster = np.zeros((k, node_num))
        for j in range(len(y_pred)):
            cluster[y_pred[j]][j] = 1.0
    else:
        cluster = np.zeros((116, 116))
        for j in range(len(y_pred)):
            for k in range(len(y_pred)):
                if y_pred[j] == y_pred[k]:
                    cluster[j][k] = 1.0

    return cluster


class myDataset(Dataset):
    def __init__(self, hfc_adj, hfc, label, edge, site_c):
        self.hfc_adj = hfc_adj
        self.hfc = hfc
        self.label = label
        self.edge = edge
        self.site = site_c
        # self.gradient = gradient
        # self.gra = gra
        # self.kendall_gra = kendall_gra

    def __getitem__(self, index):
        return self.hfc_adj[index], self.hfc[index], self.label[index], self.edge[index], self.site[index]

    def __len__(self):
        return len(self.hfc)


def training(class_model, optimizer_class, loss_class, train_iter, test_iter, target_iter, fold_i):
    centerloss = HyperbolicCenterLoss(num_classes=2, feat_dim=32).cuda()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_class, T_max=args.epoch, eta_min=5e-5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_class, T_0=10, T_mult=2, eta_min=5e-5)
    # *len(data_iter)
    train_loss = []
    # loss_domain = nn.CrossEntropyLoss()
    class_model.train()
    # discri_model.train()
    max_alpha_protoloss = 0.2  # 0.3
    max_alpha_mmdloss = 0.002
    print("--------------------start to train!--------------------")
    best_valid_acc, best_valid_sen, best_valid_spe, best_valid_f1 = 0.0, 0.0, 0.0, 0.0
    best_validtype_acc, best_validtype_sen, best_validtype_spe, best_validtype_f1 = 0.0, 0.0, 0.0, 0.0
    for e in tqdm(range(args.epoch)):
        prototype_loss_weight = min(e * 2 / args.epoch, 1.0) * max_alpha_protoloss  # 3
        mmd_loss_weight = min(e * 2 / args.epoch, 1.0) * max_alpha_mmdloss  # 3
        ASD_prototype = []
        CN_prototype = []
        tar_ASD_prototype = []
        tar_CN_prototype = []
        train_prototype = []
        train_label = []

        class_model.train()
        running_loss_c = 0
        running_loss_d = 0
        running_total_loss = 0
        train_acc = 0
        train_acc_adj = 0
        domain_acc = 0
        for _, (src_examples, tar_examples) in enumerate(zip(train_iter, target_iter)):  # 小批量读取数据
            src_FC, src_clabels, src_edge, src_info_data, src_site_c = src_examples
            tar_FC, tar_clabels, tar_edge, tar_info_data, tar_site_c = tar_examples
            tar_FC, tar_clabels, tar_edge, tar_info_data, tar_site_c = tar_FC.to(
                torch.float32).cuda(), tar_clabels.long().cuda(), tar_edge.to(torch.float32).cuda(), tar_info_data.to(
                torch.float32).cuda(), tar_site_c.to(torch.float32)
            labels = src_clabels.long().cuda()
            info_data = src_info_data.long().cuda()
            fcdata = src_FC.to(torch.float32).cuda()
            edge = src_edge.to(torch.float32).cuda()
            src_1_mmd, src_fea, y_hat, src_domain_hat = class_model(fcdata, edge, ROI_belong)  # 将数据输入网络  , fea_4center
            tar_1_mmd, tar_fea, tar_hat, tar_domain_hat = class_model(tar_FC, tar_edge, ROI_belong)
            # src_site_c = (src_site_c-1).long().cuda()
            # tar_domain_label = torch.ones(len(tar_domain_hat)).long().cuda()
            # src_domain_label = torch.zeros(len(src_domain_hat)).long().cuda()
            optimizer_class.zero_grad()
            # label = torch.argmax(labels, dim=1)
            loss_c = loss_class(y_hat, labels)  # 计算loss值
            closs = centerloss(labels, src_domain_hat)
            # loss_src_domain = loss_class(src_domain_hat, src_site_c)
            loss_mmd_1 = mmd_poincare(src_fea, tar_fea)
            loss_mmd_2 = mmd_poincare(src_1_mmd, tar_1_mmd)
            loss_mmd = mmd_poincare(src_domain_hat, tar_domain_hat)
            # loss_mcc = mcc_loss(tar_hat, temperature=2.0)
            tar_train_predict = torch.argmax(tar_hat, dim=1)
            train_bat_proto = compute_category_prototypes_tensor(src_domain_hat, labels, 2)
            tar_bat_proto = compute_category_prototypes_tensor(tar_domain_hat, tar_train_predict, 2)
            # loss_ad = loss_class(src_domain_hat, src_domain_label)+loss_class(tar_domain_hat, tar_domain_label)
            loss_proto_align = hyperbolic_prototype_loss(train_bat_proto, tar_bat_proto)
            loss_fea = loss_c +prototype_loss_weight * loss_proto_align+mmd_loss_weight * (loss_mmd + loss_mmd_1 + loss_mmd_2)  #
            # +args.lambda_closs*closs  loss_mmd_1
            # + mmd_loss_weight * (
            #                         loss_mmd + loss_mmd_2 + loss_mmd_1)
            # +args.lambda_src_domain*loss_src_domain   # +args.lambda_mcc*loss_mcc
            loss_fea.backward()  # 误差反向传播
            optimizer_class.step()  # 参数更新
            scheduler.step()
            predict = torch.argmax(y_hat, dim=1)
            # predict_adj = torch.argmax(adj_class, dim=1)
            # domain_predict = torch.argmax(domain_pred, dim=1)

            running_total_loss += loss_fea.item()
            running_loss_c += loss_c.item()  # 将每轮的loss求和
            # running_loss_d += loss_domain.item()
            equals = predict == labels
            # equals_adj = predict_adj == labels
            # equals_d = domain_predict == site_label

            train_acc += torch.mean(equals.type(torch.FloatTensor))
            # train_acc_adj += torch.mean(equals_adj.type(torch.FloatTensor))
            # domain_acc += torch.mean(equals_d.type(torch.FloatTensor))
            train_fea_type = src_domain_hat.cpu().detach().numpy()

            for sub_i in range(len(labels)):
                if labels[sub_i] == 1:
                    ASD_prototype.append(train_fea_type[sub_i])
                else:
                    CN_prototype.append(train_fea_type[sub_i])
                train_prototype.append(train_fea_type[sub_i])
                train_label.append(labels[sub_i].tolist())
        ASD_prototype = np.array(ASD_prototype)
        CN_prototype = np.array(CN_prototype)
        ASD_centers = compute_category_prototype(ASD_prototype)  # kmeans(ASD_prototype)  #
        CN_centers = compute_category_prototype(CN_prototype)

        # tar_ASD_prototype = torch.stack(tar_ASD_prototype)
        # tar_CN_prototype = torch.stack(tar_CN_prototype)
        # tar_ASD_centers = compute_category_prototype_tensor(tar_ASD_prototype)  # kmeans(ASD_prototype)  #
        # tar_CN_centers = compute_category_prototype_tensor(tar_CN_prototype)
        # proto_align_loss = 0.05*hyperbolic_prototype_loss([ASD_centers, CN_centers], [tar_ASD_centers, tar_CN_centers])
        # optimizer_class.zero_grad()  # 清空分类损失的梯度
        # proto_align_loss.backward()  # 反向传播原型对齐损失
        # optimizer_class.step()  # 更新模型参数
        train_loss.append(running_loss_c / len(train_iter))
        print("epoch: {}/{}-- ".format(e + 1, args.epoch),
              "train_loss: {:.4f}-- ".format(running_loss_c / len(train_iter)),
              "train_acc: {:.4f}--".format(train_acc / len(train_iter)),
              # "train_acc_adj: {:.4f}--".format(train_acc_adj / len(train_iter)),
              # "domain_loss: {:.4f}-- ".format(running_loss_d / len(train_iter)),
              "total_loss: {:.4f}".format(running_total_loss / len(train_iter)),
              # "proto_align_loss: {:.4f}".format(proto_align_loss*0.05)
              )
        true_label = []
        with torch.no_grad():  # 验证时不记录梯度
            class_model.eval()  # 评估模式
            labels_valid_ls = []
            predict_valid_ls = []
            predict_valid_adj = []
            valid_fea_type = []
            for fcdata_valid, labels_valid, edge_valid, info_data, site_c_test in test_iter:  # 小批量读取数据
                labels_valid = labels_valid.float().cuda()
                info_data = info_data.float().cuda()
                fcdata_valid = fcdata_valid.to(torch.float32).cuda()
                edge_valid = edge_valid.to(torch.float32).cuda()
                _2, _, y_hat, _1 = class_model(fcdata_valid, edge_valid, ROI_belong)  # 将数据输入网络  , fea_4center
                optimizer_class.zero_grad()
                predict = torch.argmax(y_hat, dim=1)
                # predict_adj = torch.argmax(adj_class, dim=1)
                # domain_predict = torch.argmax(domain_pred, dim=1)
                equals = predict == labels_valid
                # equals_d = domain_predict == site_label_valid
                labels_valid_ls.append(labels_valid.tolist())
                predict_valid_ls.append(predict.tolist())
                # predict_valid_adj.append(predict_adj.tolist())
                # train_acc += torch.mean(equals.type(torch.FloatTensor))
                true_label.append(labels_valid.tolist())
                # domain_acc += torch.mean(equals_d.type(torch.FloatTensor))
                fea_valid_type = _1.cpu().detach().numpy().reshape(-1)
                valid_fea_type.append(fea_valid_type)
            acc, senci, spec = confusion(labels_valid_ls, predict_valid_ls)
            # acc_adj, senci_adj, spec_adj = confusion(labels_valid_ls, predict_valid_adj)
            f1_s = f1_score(labels_valid_ls, predict_valid_ls)
            valid_fea_type = np.array(valid_fea_type)
            label_type = []
            for sub_i in range(len(valid_fea_type)):
                sub_i_dis = []
                asd0_dis = hyperbolic_distance_np(ASD_centers, valid_fea_type[sub_i])
                cn0_dis = hyperbolic_distance_np(CN_centers, valid_fea_type[sub_i])
                sub_i_dis.append(asd0_dis)
                sub_i_dis.append(cn0_dis)
                sub_i_dis = np.array(sub_i_dis)
                index_min = np.argmin(sub_i_dis)
                if index_min == 0:  # or index_min == 1
                    label_type.append([1])
                else:
                    label_type.append([0])
            equals_type = 0
            for sub_i in range(len(label_type)):
                if label_type[sub_i] == true_label[sub_i]:
                    equals_type = equals_type + 1
            acc_type, senci_type, spec_type = confusion(true_label, label_type)
            f1_s_type = f1_score(true_label, label_type)
            print("prototype---acc: {:.4f}...sen: {:.4f}...spec: {:.4f}...f1: {:.4f}".format(acc_type, senci_type,
                                                                                             spec_type, f1_s_type))
            # f1_adj = f1_score(labels_valid_ls, predict_valid_adj)
            # print("valid_loss: {:.4f}.. ".format(valid_loss / len(test_iter)),
            #       "valid_acc: {:.4f}".format(valid_acc / len(test_iter)))
            if acc >= best_valid_acc:
                # torch.save(class_model.state_dict(),
                #            "/SD1/luoyq/model_pth/hyperbolic_domain/UCLA_model_hlinear_fold_{}.pth".format(fold_i))
                best_valid_acc = acc
                best_valid_sen = senci
                best_valid_spe = spec
                best_valid_f1 = f1_s
            if acc_type >= best_validtype_acc:
                # torch.save(class_model.state_dict(),
                #            "/SD1/luoyq/model_pth/hyperbolic_domain/UCLA_model_prototype_fold_{}.pth".format(fold_i))
                best_validtype_acc = acc_type
                best_validtype_sen = senci_type
                best_validtype_spe = spec_type
                best_validtype_f1 = f1_s_type
            print("hyplinear---acc: {:.4f}...sen: {:.4f}...spec: {:.4f}...f1: {:.4f}".format(acc, senci, spec, f1_s))
            # print("adj: acc: {:.4f}...sen: {:.4f}...spec: {:.4f}...f1: {:.4f}".format(acc_adj, senci_adj, spec_adj,
            #                                                                           f1_adj))
            if e == args.epoch - 1:
                total_valid_sen.append(senci)
                total_valid_f1.append(f1_s)
                total_valid_spe.append(spec)
                total_valid_acc.append(acc)
        # "center loss: {:.4f}------".format(running_closs/len(data_iter)),
        #               "total_loss: {:.4f}".format(running_total_loss/len(data_iter))
    # torch.save(class_model.state_dict(), "/SD1/luoyq/model_pth/MSB_Net/model_data_{}.pth".format(i))
    # plt.figure(figsize=(15, 10), dpi=100)
    # plt.plot(range(1, args.epoch + 1), train_loss, color='r', linestyle='--', label='loss')
    # plt.title("Train loss vs epoch-{}".format(args.epoch), fontsize=30)
    # plt.xlabel("epochs", fontsize=30)
    # plt.tick_params(labelsize=15)
    # plt.ylabel("Train loss", fontsize=30)
    # plt.show()
    total_valid_sen_adj.append(best_valid_sen)
    total_valid_f1_adj.append(best_valid_f1)
    total_valid_spe_adj.append(best_valid_spe)
    total_valid_acc_adj.append(best_valid_acc)

    total_validtype_senci.append(best_validtype_sen)
    total_validtype_spec.append(best_validtype_spe)
    total_validtype_f1.append(best_validtype_f1)
    total_validtype_acc.append(best_validtype_acc)
    # valid_fea_type = np.array(valid_fea_type)

    valid_loss = 0
    valid_acc = 0


def setup_seed(seed):
    """ 该方法用于固定随机数

    Args:
        seed: 随机种子数

    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def split_indices(indices, n_splits, fold):
    """
    根据 fold 对指定 indices 进行 KFold 划分
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(kf.split(indices))
    train_idx, test_idx = splits[fold]
    return indices[train_idx], indices[test_idx]


def load_domain_data(dict_4domain, edge, fold):
    """
        加载不同站点的训练集、测试集和目标域数据
    """
    # 提取数据
    FC_data = dict_4domain["fc"]
    site_labels = dict_4domain["site_labels"]
    class_labels = dict_4domain["label"]
    info_data = dict_4domain["info"]
    # edge = dict_4domain["fc"]
    # 获取各站点索引
    indices_by_site = {site: np.where(site_labels == site)[0] for site in range(4)}

    # 目标域站点索引
    target_indices = indices_by_site[0]
    train_t, test_t = split_indices(target_indices, n_splits=5, fold=fold)

    # # 其他站点划分
    # train_s, test_s = [], []
    # for site in range(1, 4):
    #     train, test = split_indices(indices_by_site[site], n_splits=5, fold=fold)
    #     train_s.append(train)
    #     test_s.append(test)

    # 合并训练集和测试集
    train_indices = list(indices_by_site[1]) + list(indices_by_site[2]) + list(indices_by_site[3])

    # 用于每个站点取1/5作为测试集
    # test_t = np.concatenate([test_t] + test_s)
    # train_indices = np.concatenate(train_s)

    # 提取训练集、目标域和测试集
    FC_train, class_labels_train, edge_train, info_data_train, site_labels_train = (
        FC_data[train_indices], class_labels[train_indices],
        edge[train_indices], info_data[train_indices], site_labels[train_indices])
    # edge_mask = F_score_mask(FC_train, class_labels_train)
    # edge_train = edge_train*edge_mask

    FC_target, class_labels_target, edge_target, info_data_target, site_labels_target = (
    FC_data[train_t], class_labels[train_t],
    edge[train_t], info_data[train_t], site_labels[train_t])
    FC_test, class_labels_test, edge_test, info_data_test, site_labels_test = (FC_data[test_t], class_labels[test_t],
                                                                               edge[test_t], info_data[test_t],
                                                                               site_labels[test_t])

    # edge_target = edge_target * edge_mask
    # edge_test = edge_test * edge_mask
    train_data = myDataset(FC_train, class_labels_train, edge_train, info_data_train, site_labels_train)
    test_data = myDataset(FC_test, class_labels_test, edge_test, info_data_test, site_labels_test)
    target_data = myDataset(FC_target, class_labels_target, edge_target, info_data_target, site_labels_target)
    train_iter = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_iter = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    target_iter = DataLoader(target_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_iter, test_iter, target_iter


def main(args, ROI_belong):
    setup_seed(args.seed)
    # FC_data = np.load(r"./data/FC.npy")
    # site_labels = np.load(r"./data/site_label.npy")
    # class_labels = np.load(r"./data/class_label.npy")
    # edge = np.load("/SD1/luoyq/shuffle_good/Fscore_edge.npy")
    dict_4domain = np.load("/SD1/luoyq/ABIDE_site_data/dict_4domain.npy", allow_pickle=True).item()
    FC_data = dict_4domain["fc"]
    site_labels = dict_4domain["site_labels"]
    class_labels = dict_4domain["label"]
    info_data = dict_4domain["info"]
    edge = np.load(r"/SD1/luoyq/ABIDE_site_data/edge_indicidual_10.npy")  # edge_indicidual_15.npy   KNN_hyper_15%

    k_cluster = 7
    # Fc_4comm = FC_data[:, 1:108:2, 1:108:2]
    cluster_gra = gradient_avg_mask(FC_data, k_cluster, mask=False)
    node_num4comm = []
    for community_i in range(len(cluster_gra)):
        ROI_belong[community_i] = torch.tensor(np.where(cluster_gra[community_i] == 1)[0]).cuda()
        node_num4comm.append(len(torch.tensor(np.where(cluster_gra[community_i] == 1)[0])))
    node_num4comm = torch.tensor(node_num4comm)
    cluster_gra = torch.tensor(cluster_gra).to(torch.float32).cuda()

    for fold_count in range(5):
        print("fold: {}".format(fold_count + 1))
        train_iter, test_iter, target_iter = load_domain_data(dict_4domain, edge, fold_count)
        target_iter = cycle(target_iter)
        # federated setup
        # class_model = feature_extractor(node_num4comm).to(device)
        class_model = HyperbolicDomainAdaptLayer(node_num4comm, k=k_cluster).cuda()  # feature_extractor(node_num4comm).to(device)
        optimizer_class = geoopt.optim.RiemannianAdam(class_model.parameters(), lr=0.0001)
        loss_class = nn.CrossEntropyLoss()  # FocalLoss()
        # discri_model = AdversarialNetworkSp().to(device)
        # optimizer_discrimination = optim.Adam(discri_model.parameters(), lr=0.0001)
        # loss_discrimination = nn.CrossEntropyLoss()
        training(class_model, optimizer_class, loss_class, train_iter, test_iter, target_iter, fold_count)

        del class_model
        # del discri_model
        del optimizer_class
        # del optimizer_discrimination
        del loss_class
        # del loss_discrimination


# ==========================================================================
if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    # torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    global ROI_belong
    ROI_belong = {}
    global total_valid_sen, total_valid_sen_adj
    total_valid_sen, total_valid_sen_adj = [], []
    global total_valid_spe, total_valid_spe_adj
    total_valid_spe, total_valid_spe_adj = [], []
    global total_valid_f1, total_valid_f1_adj
    total_valid_f1, total_valid_f1_adj = [], []
    global total_valid_acc, total_valid_acc_adj
    total_valid_acc, total_valid_acc_adj = [], []

    global total_validtype_senci, total_validtype_spec, total_validtype_f1, total_validtype_acc
    total_validtype_senci, total_validtype_spec, total_validtype_f1, total_validtype_acc = [], [], [], []

    global total_valid_fusion_sen, total_valid_fusion_spe, total_valid_fusion_f1, total_valid_fusion_acc
    total_valid_fusion_sen, total_valid_fusion_spe, total_valid_fusion_f1, total_valid_fusion_acc = [], [], [], []

    # specify for dataset site
    parser.add_argument('--split', type=int, default=1, help='select 0-4 fold')
    # do not need to change
    parser.add_argument('--pace', type=int, default=50, help='communication pace')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--type', type=str, default='G', help='Gaussian or Lap')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_mmd', type=float, default=0.002)
    parser.add_argument('--lambda_mcc', type=float, default=0.0)
    parser.add_argument('--lambda_l1', type=float, default=1e-4)
    parser.add_argument('--lambda_src_domain', type=float, default=0.2)
    parser.add_argument('--lambda_adloss', type=float, default=0.3)
    parser.add_argument('--lambda_closs', type=float, default=0.02)
    parser.add_argument('--lambda_protoalign', type=float, default=0.2, help='the weight of prototype align')
    parser.add_argument('--nsteps', type=int, default=100, help='training steps/epoach')
    parser.add_argument('-tbs1', '--test_batch_size1', type=int, default=145, help='NYU test batch size')
    parser.add_argument('-tbs2', '--test_batch_size2', type=int, default=265, help='UM test batch size')
    parser.add_argument('-tbs3', '--test_batch_size3', type=int, default=205, help='USM test batch size')
    parser.add_argument('-tbs4', '--test_batch_size4', type=int, default=85, help='UCLA test batch size')
    parser.add_argument('--overlap', type=bool, default=True, help='augmentation method')
    parser.add_argument('--sepnorm', type=bool, default=False, help='normalization method')
    parser.add_argument('--id_dir', type=str, default='./idx')
    parser.add_argument('--res_dir', type=str, default='./result/align_overlap')
    parser.add_argument('--vec_dir', type=str, default='')  # ./data/HO_vector_overlap
    parser.add_argument('--model_dir', type=str, default='./model/align_overlap')

    args = parser.parse_args()
    assert args.split in [0, 1, 2, 3, 4]
    print("7 comm 2gradient")
    # cluster_gra = gradient_avg_mask(FC_aal, k_cluster, mask=False)
    ROI_belong = {}
    main(args, ROI_belong)
    print(
        "valid_acc: {:.4f}...valid_sen: {:.4f}...valid_spec: {:.4f}...valid_f1: {:.4f}".format(
            torch.mean(torch.tensor(np.array(total_valid_acc))),
            torch.mean(torch.tensor(np.array(total_valid_sen))),
            torch.mean(torch.tensor(np.array(total_valid_spe))),
            torch.mean(torch.tensor(np.array(total_valid_f1)))))
    print(
        "valid_acc_best: {:.4f}...valid_sen_best: {:.4f}...valid_spec_best: {:.4f}...valid_f1_best: {:.4f}".format(
            torch.mean(torch.tensor(np.array(total_valid_acc_adj))),
            torch.mean(torch.tensor(np.array(total_valid_sen_adj))),
            torch.mean(torch.tensor(np.array(total_valid_spe_adj))),
            torch.mean(torch.tensor(np.array(total_valid_f1_adj)))))

    print(
        "validtype_acc_best: {:.4f}...validtype_sen_best: {:.4f}...validtype_spec_best: {:.4f}...validtype_f1_best: {:.4f}".format(
            torch.mean(torch.tensor(np.array(total_validtype_acc))),
            torch.mean(torch.tensor(np.array(total_validtype_senci))),
            torch.mean(torch.tensor(np.array(total_validtype_spec))),
            torch.mean(torch.tensor(np.array(total_validtype_f1)))))

    # print(
    #     "tar_fusion---valid_acc_best: {:.4f}...valid_sen_best: {:.4f}...valid_spec_best: {:.4f}...valid_f1_best: {:.4f}".format(
    #         torch.mean(torch.tensor(np.array(total_valid_fusion_acc))),
    #         torch.mean(torch.tensor(np.array(total_valid_fusion_sen))),
    #         torch.mean(torch.tensor(np.array(total_valid_fusion_spe))),
    #         torch.mean(torch.tensor(np.array(total_valid_fusion_f1)))))
