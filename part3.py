import csv, os
import numpy as np

with open("data/records.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    data = [row for row in reader]

X_raw = np.array([[float(v) for v in r[1:6]] for r in data])
outcome = np.array([int(r[8]) for r in data])
value = np.array([float(r[9]) for r in data])
X_std = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0, ddof=1)
res = {}

# 3.1 K-means from scratch
def run_kmeans(K):
    centroids = X_std[[i * (800 // K) for i in range(K)]].copy()
    labels = np.zeros(800, dtype=int)
    while True:
        dists = np.sum((X_std[:, None, :] - centroids[None, :, :])**2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(labels, new_labels): break
        labels = new_labels
        for k in range(K):
            if np.sum(labels == k) > 0:
                centroids[k] = np.mean(X_std[labels == k], axis=0)
    wcss = np.sum((X_std - centroids[labels])**2)
    return wcss, labels, centroids

wcss_vals = {}
for k in range(2, 7):
    wcss, lbls, cents = run_kmeans(k)
    res[f'wcss_k{k}'] = wcss
    wcss_vals[k] = wcss
    if k == 3: labels_k3, cents_k3 = lbls, cents

# 3.2 Choose K with elbow
drops = {k: wcss_vals[k] - wcss_vals[k+1] for k in range(2, 6)}
elbow_k = 3
for k in [3, 4, 5]:
    if drops[k] < 0.5 * drops[k-1]:
        elbow_k = k; break
res['elbow_k'] = elbow_k

# 3.3 Profile K=3 segments
means_X3 = [np.mean(X_raw[labels_k3 == k][:, 2]) for k in range(3)]
order = np.argsort(means_X3)
seg_names = ['A', 'B', 'C']

with open("outputs/part3_segments.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['segment','size','mean_X1','mean_X2','mean_X3','mean_X4','mean_X5','outcome_rate','mean_value'])
    for i, orig_k in enumerate(order):
        idx = (labels_k3 == orig_k)
        size = np.sum(idx)
        res[f'size_segment_{seg_names[i]}'] = size
        means = np.mean(X_raw[idx], axis=0)
        out_rate = np.mean(outcome[idx])
        val_mean = np.mean(value[idx])
        writer.writerow([seg_names[i], size, f"{means[0]:.4f}", f"{means[1]:.4f}", f"{means[2]:.4f}", 
                         f"{means[3]:.4f}", f"{means[4]:.4f}", f"{out_rate:.4f}", f"{val_mean:.4f}"])

# 3.4 Hierarchical clustering
D_init = np.array([
    [0, 7.9, 7.4, 2.9, 2.9, 6.1], [7.9, 0, 1.4, 5.9, 5.3, 2.0], [7.4, 1.4, 0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0, 2.8, 4.5], [2.9, 5.3, 4.6, 2.8, 0, 3.4], [6.1, 2.0, 1.3, 4.5, 3.4, 0]
])
entities = [f"E{i+1}" for i in range(6)]

def hclust(method):
    D = D_init.copy()
    D = D.astype(float)
    np.fill_diagonal(D, np.inf)
    clusters = [[e] for e in entities]
    merges, heights = [], []
    for _ in range(5):
        min_idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = min(min_idx), max(min_idx)
        heights.append(D[i, j])
        c1, c2 = clusters[i], clusters[j]
        new_c = sorted(c1 + c2)
        ca = c1 if c1[0] < c2[0] else c2
        cb = c2 if c1[0] < c2[0] else c1
        merges.append((ca, cb, D[i, j]))
        clusters.append(new_c)
        new_dists = []
        for k in range(len(D)):
            if k != i and k != j:
                if method == 'single': d = min(D[i, k], D[j, k])
                elif method == 'complete': d = max(D[i, k], D[j, k])
                else: d = (D[i, k] * len(clusters[i]) + D[j, k] * len(clusters[j])) / (len(clusters[i]) + len(clusters[j]))
                new_dists.append(d)
        D = np.delete(D, [i, j], axis=0)
        D = np.delete(D, [i, j], axis=1)
        new_row = np.array(new_dists + [np.inf])
        if len(D) > 0:
            D = np.vstack([D, new_dists])
            D = np.column_stack([D, new_row])
        for idx in sorted([i, j], reverse=True): del clusters[idx]
    
    gaps = [heights[i] - heights[i-1] for i in range(1, 5)]
    cut_idx = np.argmax(gaps)
    return merges, 6 - (cut_idx + 1)

merges_s, cut_s = hclust('single')
merges_c, cut_c = hclust('complete')
merges_a, cut_a = hclust('average')
res['single_clusters_at_cut'] = cut_s
res['complete_clusters_at_cut'] = cut_c
res['average_clusters_at_cut'] = cut_a

with open("outputs/part3_hierarchical.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['linkage','merge_no','cluster_a','cluster_b','height'])
    for m, m_data in zip(['single', 'complete', 'average'], [merges_s, merges_c, merges_a]):
        for step, (ca, cb, h) in enumerate(m_data):
            writer.writerow([m, step+1, "+".join(ca), "+".join(cb), f"{h:.4f}"])

# 3.5 PCA
S = X_std.T @ X_std / 799
vals, vecs = np.linalg.eigh(S)
idx = np.argsort(vals)[::-1]
vals, vecs = vals[idx], vecs[:, idx]

for i in range(5):
    max_idx = np.argmax(np.abs(vecs[:, i]))
    if vecs[max_idx, i] < 0: vecs[:, i] *= -1

res['eigenvalue_sum'] = np.sum(vals)
for i in range(5): res[f'eigenvalue_{i+1}'] = vals[i]
for i in range(5): res[f'explained_ratio_{i+1}'] = vals[i] / np.sum(vals)
for i in range(5): res[f'cumulative_ratio_{i+1}'] = np.sum(vals[:i+1]) / np.sum(vals)

cum = np.cumsum(vals) / np.sum(vals)
res['components_for_80pct'] = int(np.argmax(cum >= 0.8) + 1)

for i in range(5): res[f'pc1_loading_X{i+1}'] = vecs[i, 0]
for i in range(5): res[f'pc2_loading_X{i+1}'] = vecs[i, 1]

Z = X_std @ vecs[:, :2]
for i in range(3):
    res[f'row{i+1}_pc1'] = Z[i, 0]
    res[f'row{i+1}_pc2'] = Z[i, 1]

for i, orig_k in enumerate(order):
    res[f'mean_pc1_segment_{seg_names[i]}'] = np.mean(Z[labels_k3 == orig_k][:, 0])

with open("outputs/part3_results.txt", "w") as f:
    for k, v in res.items():
        if isinstance(v, float): f.write(f"{k} = {v:.4f}\n")
        else: f.write(f"{k} = {v}\n")