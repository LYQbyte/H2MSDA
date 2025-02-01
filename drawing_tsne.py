import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from manifolds.hyp_layers_v3 import HyperbolicDomainAdaptLayer
from sklearn.cluster import KMeans
import torch
from training_4domain import myDataset
from torch.utils.data import DataLoader


def gradient_avg_mask(Fc_all, k, mask=False, node_num=116):
    # Fc_all shape: num 116 116
    # 聚类为k个簇
    # avg_Fc = np.sum(Fc_all, axis=0)/len(Fc_all)
    # gm1 = GradientMaps(n_components=2,  approach='dm', kernel='cosine', random_state=42)
    # gm1.fit(avg_Fc)
    # gradient = gm1.gradients_
    gradient = np.load("/SD1/luoyq/shuffle_good/gradient_hyper_spear.npy")[:, 0:2]
    # gradient = np.load("/SD1/luoyq/ABIDE_site_data/site_data_gradient_hyper.npy")[:, 0:2]
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


dict_4domain = np.load("/SD1/luoyq/ABIDE_site_data/dict_4domain.npy", allow_pickle=True).item()

FC_data = dict_4domain["fc"]
site_labels = dict_4domain["site_labels"]
class_label = dict_4domain["label"]
edge = np.load(r"/SD1/luoyq/ABIDE_site_data/edge_indicidual_10.npy")
k_cluster = 7
# Fc_4comm = FC_data[:, 1:108:2, 1:108:2]
ROI_belong = {}
cluster_gra = gradient_avg_mask(FC_data, k_cluster, mask=False)
node_num4comm = []
for community_i in range(len(cluster_gra)):
    ROI_belong[community_i] = torch.tensor(np.where(cluster_gra[community_i] == 1)[0]).cuda()
    node_num4comm.append(len(torch.tensor(np.where(cluster_gra[community_i] == 1)[0])))
node_num4comm = torch.tensor(node_num4comm)

class_model = HyperbolicDomainAdaptLayer(node_num4comm).cuda()

checkpoint = torch.load(r'/SD1/luoyq/model_pth/hyperbolic_domain/UCLA_model_prototype_fold_4.pth')

# 将加载的权重应用到模型
class_model.load_state_dict(checkpoint)

class_model.eval()

train_data = myDataset(FC_data, class_label, edge, class_label, site_labels)
train_iter = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=4)

total_fea = []
total_site_c = []
for src_FC, src_clabels, src_edge, src_info_data, src_site_c in train_iter:
    src_edge = src_edge.to(torch.float32).cuda()
    src_FC = src_FC.to(torch.float32).cuda()
    total_site_c.append(src_site_c)
    src_1_mmd, src_fea, y_hat, src_domain_hat = class_model(src_FC, src_edge, ROI_belong)
    total_fea.append(src_domain_hat.cpu().detach().numpy())

# np.save(r"/SD1/luoyq/ABIDE_site_data/TSNE_aligned.npy", np.array(total_fea))
# np.save(r"/SD1/luoyq/ABIDE_site_data/site_label_tsne.npy", np.array(total_site_c))

#
# plt.rc('font', family='Times New Roman', size=12)
# tsne = TSNE(n_components=2, random_state=42, perplexity=30)
#
# scaler = StandardScaler()
# X_raw = scaler.fit_transform(upper_triangle_data)
#
#
# site_names = {0: "NYU", 1: "UM", 2: "USM", 3: "UCLA"}
# y_labels = [site_names[label] for label in site_labels]  # 将数值标签映射为站点名称
# # 对原始数据进行 t-SNE 降维
# X_raw_tsne = tsne.fit_transform(X_raw)
#
# df_raw = pd.DataFrame(X_raw_tsne, columns=["Dim1", "Dim2"])
#
# df_raw["Site"] = y_labels
#
#
#
# sns.set(style="whitegrid", context="talk")
# # 绘制 t-SNE 可视化
# sns.set(style="white")
# plt.figure(figsize=(7, 6), dpi=100)
# # plt.subplot(1, 2, 1)
# sns.scatterplot(
#     x="Dim1",
#     y="Dim2",
#     hue="Site",
#     palette="tab10",  # 使用 seaborn 的调色盘
#     data=df_raw,
#     s=60,  # 点大小
#     edgecolor="black",
#     alpha=0.9
# )
# plt.title("Original Brain Networks",fontsize=12, fontweight='bold')
# # plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.xlabel("")
# plt.ylabel("")
#
# plt.tight_layout()
# plt.show()

