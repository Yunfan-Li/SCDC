import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics
from munkres import Munkres
from sklearn.cluster import KMeans


def inference(net, test_dataloader):
    net.eval()
    feature_vec, type_vec, batch_vec, pred_vec, soft_vec = [], [], [], [], []
    for (x, sf, rc, t, b), index in test_dataloader:
        x = x.cuda()
        b = b.cuda()
        with torch.no_grad():
            b = b.cuda()
            with torch.no_grad():
                t_, c = net.encode_feature_cluster(x)
            c = torch.nn.functional.softmax(c, dim=1)
            c.detach_()
            pred = torch.argmax(c, dim=1)
        feature_vec.extend(t_.cpu().numpy())
        type_vec.extend(t.cpu().numpy())
        batch_vec.extend(b.cpu().numpy())
        pred_vec.extend(pred.cpu().numpy())
        soft_vec.extend(c.cpu().numpy())
    feature_vec, type_vec, batch_vec, pred_vec, soft_vec = np.array(
        feature_vec), np.array(type_vec), np.array(batch_vec), np.array(
            pred_vec), np.array(soft_vec)

    return feature_vec, type_vec, batch_vec, pred_vec, soft_vec


def kmeans_result(feature_vec,
                  type_vec,
                  batch_vec,
                  n_clusters,
                  batch_metric=False):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(feature_vec)
    nmi, ari, acc = cluster_metrics(type_vec, kmeans.labels_)
    tqdm.write('NMI=%.4f, ACC=%.4f, ARI=%.4f' % (nmi, acc, ari), end='')
    if batch_metric:
        lisi = batch_effect_metrics(feature_vec, batch_vec)
        tqdm.write(', LISI=%.4f' % lisi, end='')
    tqdm.write('')
    return kmeans


def evaluate(feature_vec,
             pred_vec,
             type_vec,
             batch_vec,
             soft_vec,
             batch_metric=False):
    if len(set(pred_vec)) != len(set(type_vec)):
        kmeans = KMeans(n_clusters=len(set(type_vec))).fit(soft_vec)
        pred_vec = kmeans.labels_
    tqdm.write("Evaluating the clustering results...")
    nmi, ari, acc = cluster_metrics(type_vec, pred_vec)
    tqdm.write('NMI=%.4f, ACC=%.4f, ARI=%.4f' % (nmi, acc, ari), end='')
    if batch_metric:
        ber, ari_b = batch_effect_metrics(feature_vec, batch_vec, pred_vec,
                                          type_vec)
        tqdm.write(', BER=%.4f, ARI_b=%.4f' % (ber, ari_b), end='')
    tqdm.write('')
    return pred_vec


def cluster_metrics(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(label, pred_adjusted)
    # acc = 0
    return nmi, ari, acc


def batch_effect_metrics(feature_vec, batch_vec, pred_vec, type_vec):
    ber, single_types = compute_ber(batch_vec, pred_vec, type_vec)

    for t in single_types:
        idx = (type_vec != t)
        batch_vec = batch_vec[idx]
        pred_vec = pred_vec[idx]
        type_vec = type_vec[idx]

    ari_b = metrics.adjusted_rand_score(batch_vec, pred_vec)
    return ber, ari_b


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    confusion_matrix = metrics.confusion_matrix(y_true,
                                                cluster_assignments,
                                                labels=None)
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    pred_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = pred_to_true_cluster_labels[cluster_assignments]
    return y_pred


def compute_ber(batch_vec, pred_vec, type_vec):
    n_cell, n_batch, n_type = batch_vec.shape[0], np.max(
        batch_vec) + 1, np.max(type_vec) + 1

    O = torch.ones((n_type, n_batch)).cuda()
    E = torch.ones((n_type, n_batch)).cuda()

    for i in range(n_cell):
        O[pred_vec[i], batch_vec[i]] += 1
        E[type_vec[i], batch_vec[i]] += 1

    cost_matrix = torch.zeros((n_type, n_type)).cuda()
    for i in range(n_type):
        for j in range(n_type):
            cost_matrix[j,
                        i] = torch.mul(O[i],
                                       torch.log(O[i] / E[j])).mean().item()
    indices = Munkres().compute(cost_matrix)
    pred_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    O = O[pred_to_true_cluster_labels]

    ber = torch.mul(O, torch.log(O / E)).mean().item()

    single_types = []
    for i in range(n_type):
        batch_num = (E[i] > 1).sum()
        if batch_num == 1:
            single_types.append(i)

    return ber, single_types
