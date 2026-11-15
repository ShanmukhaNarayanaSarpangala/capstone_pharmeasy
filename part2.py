import csv, math
import numpy as np

# Load Data
with open("data/records.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    data = [row for row in reader]

y = np.array([int(r[8]) for r in data])
X = np.array([[float(v) for v in r[1:8]] for r in data])

# 2.1 Split and standardise
idx_0 = np.where(y == 0)[0]
idx_1 = np.where(y == 1)[0]
train_0_count = int(7 * len(idx_0) / 10)
train_1_count = int(7 * len(idx_1) / 10)

train_idx = np.sort(np.concatenate([idx_0[:train_0_count], idx_1[:train_1_count]]))
test_idx = np.sort(np.concatenate([idx_0[train_0_count:], idx_1[train_1_count:]]))

X_train_raw, y_train = X[train_idx], y[train_idx]
X_test_raw, y_test = X[test_idx], y[test_idx]

train_mean = np.mean(X_train_raw, axis=0)
train_sd = np.std(X_train_raw, axis=0, ddof=1)

X_train = (X_train_raw - train_mean) / train_sd
X_test = (X_test_raw - train_mean) / train_sd

res = {
    'train_rows': len(train_idx),
    'train_positives': np.sum(y_train),
    'test_rows': len(test_idx),
    'test_positives': np.sum(y_test)
}

# 2.2 Logistic regression from scratch
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def get_loss(w, b, X_data, y_data):
    y_hat = np.clip(sigmoid(X_data @ w + b), 1e-15, 1 - 1e-15)
    return -np.mean(y_data * np.log(y_hat) + (1 - y_data) * np.log(1 - y_hat))

w, b, eta = np.zeros(7), 0.0, 0.1
res['loss_epoch_0'] = get_loss(w, b, X_train, y_train)

for epoch in range(1, 1001):
    y_hat = sigmoid(X_train @ w + b)
    w -= eta * np.mean((y_hat - y_train)[:, None] * X_train, axis=0)
    b -= eta * np.mean(y_hat - y_train)
    if epoch in [1, 10, 100, 1000]:
        res[f'loss_epoch_{epoch}'] = get_loss(w, b, X_train, y_train)

for i in range(7): res[f'weight_X{i+1}'] = w[i]
res['bias'] = b
res['test_bce'] = get_loss(w, b, X_test, y_test)

# 2.3 Learning-rate effect
for lr in [0.01, 0.1, 1.0]:
    w_lr, b_lr = np.zeros(7), 0.0
    for _ in range(200):
        y_hat_lr = sigmoid(X_train @ w_lr + b_lr)
        w_lr -= lr * np.mean((y_hat_lr - y_train)[:, None] * X_train, axis=0)
        b_lr -= lr * np.mean(y_hat_lr - y_train)
    res[f'loss_epoch_200_eta_{lr}'] = get_loss(w_lr, b_lr, X_train, y_train)

# 2.4 Odds ratios
for i in range(7): res[f'odds_ratio_X{i+1}'] = np.exp(w[i])

# 2.5 Evaluate on test rows
y_test_prob = sigmoid(X_test @ w + b)

def eval_metrics(y_true, y_prob, tau, prefix):
    y_pred = (y_prob >= tau).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    res[f'{prefix}_tp'], res[f'{prefix}_fp'], res[f'{prefix}_tn'], res[f'{prefix}_fn'] = tp, fp, tn, fn
    res[f'{prefix}_accuracy'] = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    res[f'{prefix}_precision'] = prec
    res[f'{prefix}_recall'] = rec
    res[f'{prefix}_f1'] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    res[f'{prefix}_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

eval_metrics(y_test, y_test_prob, 0.5, 'tau_0.5')
eval_metrics(y_test, y_test_prob, 0.3, 'tau_0.3')
res['blind_accuracy'] = np.sum(y_test == 0) / len(y_test)

# 2.6 Threshold sweep
with open("outputs/part2_thresholds.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['tau','tp','fp','tn','fn','tpr','fpr'])
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred = (y_test_prob >= tau).astype(int)
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        writer.writerow([f"{tau:.1f}", tp, fp, tn, fn, f"{tpr:.4f}", f"{fpr:.4f}"])

# 2.7 KNN with CV
def get_dists(query, refs): return np.sqrt(np.sum((refs - query)**2, axis=1))

best_k, best_f1, knn_cv = 1, -1, []
for K in range(1, 22, 2):
    fold_f1s = []
    for fold in range(5):
        v_idx = [i for i in range(len(train_idx)) if i % 5 == fold]
        t_idx = [i for i in range(len(train_idx)) if i % 5 != fold]
        
        y_p = []
        for query in X_train[v_idx]:
            dists = get_dists(query, X_train[t_idx])
            top_k_y = y_train[t_idx][np.argsort(dists, kind='stable')[:K]]
            y_p.append(1 if np.sum(top_k_y) >= (K + 1) / 2 else 0)
            
        y_p = np.array(y_p)
        y_v = y_train[v_idx]
        tp = np.sum((y_p == 1) & (y_v == 1))
        fp = np.sum((y_p == 1) & (y_v == 0))
        fn = np.sum((y_p == 0) & (y_v == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        fold_f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0)
        
    mean_f1 = np.mean(fold_f1s)
    knn_cv.append((K, mean_f1))
    if mean_f1 > best_f1:
        best_f1, best_k = mean_f1, K

with open("outputs/part2_knn_cv.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['K','mean_cv_f1'])
    for K, mf1 in knn_cv: writer.writerow([K, f"{mf1:.4f}"])

res['knn_best_k'] = best_k

# KNN Test
y_test_pred_knn = []
for query in X_test:
    dists = get_dists(query, X_train)
    top_k_y = y_train[np.argsort(dists, kind='stable')[:best_k]]
    y_test_pred_knn.append(1 if np.sum(top_k_y) >= (best_k + 1) / 2 else 0)

y_test_pred_knn = np.array(y_test_pred_knn)
res['knn_tp'] = np.sum((y_test_pred_knn == 1) & (y_test == 1))
res['knn_fp'] = np.sum((y_test_pred_knn == 1) & (y_test == 0))
res['knn_tn'] = np.sum((y_test_pred_knn == 0) & (y_test == 0))
res['knn_fn'] = np.sum((y_test_pred_knn == 0) & (y_test == 1))
res['knn_accuracy'] = (res['knn_tp'] + res['knn_tn']) / len(y_test)
prec_k = res['knn_tp'] / (res['knn_tp'] + res['knn_fp']) if (res['knn_tp'] + res['knn_fp']) > 0 else 0
rec_k = res['knn_tp'] / (res['knn_tp'] + res['knn_fn']) if (res['knn_tp'] + res['knn_fn']) > 0 else 0
res['knn_precision'] = prec_k
res['knn_recall'] = rec_k
res['knn_f1'] = 2 * prec_k * rec_k / (prec_k + rec_k) if (prec_k + rec_k) > 0 else 0

# 2.8 Choose alert threshold with money
y_train_prob = sigmoid(X_train @ w + b)
best_tau, best_cost = -1, float('inf')

for k in range(1, 20):
    tau = k / 20.0
    y_p = (y_train_prob >= tau).astype(int)
    cost = 50 * np.sum((y_p == 1) & (y_train == 0)) + 400 * np.sum((y_p == 0) & (y_train == 1))
    if cost < best_cost:
        best_cost, best_tau = cost, tau

res['cost_tau'], res['cost_train_at_tau'] = best_tau, int(best_cost)

def get_cost(y_t, y_p_prob, tau):
    y_p = (y_p_prob >= tau).astype(int)
    return int(50 * np.sum((y_p == 1) & (y_t == 0)) + 400 * np.sum((y_p == 0) & (y_t == 1)))

res['cost_test_at_tau'] = get_cost(y_test, y_test_prob, best_tau)
res['cost_test_at_0.5'] = get_cost(y_test, y_test_prob, 0.5)
res['cost_test_flag_nobody'] = int(400 * np.sum(y_test == 1))

# Write Output
with open("outputs/part2_results.txt", "w") as f:
    for k, v in res.items():
        if isinstance(v, float): f.write(f"{k} = {v:.4f}\n")
        else: f.write(f"{k} = {v}\n")