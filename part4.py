import csv, math
import numpy as np

with open("data/records.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    data = [row for row in reader]

y = np.array([int(r[8]) for r in data])
X_raw = np.array([[float(v) for v in r[1:8]] for r in data])
idx_0, idx_1 = np.where(y == 0)[0], np.where(y == 1)[0]
t0, t1 = int(7 * len(idx_0) / 10), int(7 * len(idx_1) / 10)
train_idx = np.sort(np.concatenate([idx_0[:t0], idx_1[:t1]]))
test_idx = np.sort(np.concatenate([idx_0[t0:], idx_1[t1:]]))

X_train, y_train = X_raw[train_idx], y[train_idx]
X_test, y_test = X_raw[test_idx], y[test_idx]

res = {}

# 4.1 Splits
def get_entropy(y_node):
    if len(y_node) == 0: return 0
    p = np.mean(y_node)
    if p == 0 or p == 1: return 0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def get_gini(y_node):
    if len(y_node) == 0: return 0
    p = np.mean(y_node)
    return 1 - (p**2 + (1 - p)**2)

root_ent = get_entropy(y_train)
root_gini = get_gini(y_train)
res['root_entropy'] = root_ent
res['root_gini'] = root_gini

splits = {
    'S1': X_train[:, 5] == 1,
    'S2': X_train[:, 6] == 1,
    'S3': X_train[:, 2] > 0.5,
    'S4': X_train[:, 0] <= 24
}

split_data = []
best_ig, best_split_ig = -1, ''
best_gg, best_split_gg = -1, ''

for name, mask in splits.items():
    y_yes, y_no = y_train[mask], y_train[~mask]
    n_y, n_n = len(y_yes), len(y_no)
    w_y, w_n = n_y / len(y_train), n_n / len(y_train)
    ig = root_ent - (w_y * get_entropy(y_yes) + w_n * get_entropy(y_no))
    gg = root_gini - (w_y * get_gini(y_yes) + w_n * get_gini(y_no))
    split_data.append([name, n_y, np.sum(y_yes), n_n, np.sum(y_no), f"{ig:.4f}", f"{gg:.4f}"])
    if ig > best_ig: best_ig, best_split_ig = ig, name
    if gg > best_gg: best_gg, best_split_gg = gg, name

res['best_split_by_info_gain'] = best_split_ig
res['best_split_by_gini_gain'] = best_split_gg

with open("outputs/part4_splits.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['split','n_yes','positives_yes','n_no','positives_no','info_gain','gini_gain'])
    writer.writerows(split_data)

# 4.2 Voting Models
mask = splits[best_split_ig]
stump_yes_prob = np.mean(y_train[mask])
stump_no_prob = np.mean(y_train[~mask])
res['stump_rate_yes'] = stump_yes_prob
res['stump_rate_no'] = stump_no_prob

# Rebuild log/knn quickly for voting
t_m, t_s = np.mean(X_train, axis=0), np.std(X_train, axis=0, ddof=1)
X_train_s, X_test_s = (X_train - t_m) / t_s, (X_test - t_m) / t_s

w, b, lr = np.zeros(7), 0.0, 0.1
for _ in range(1000):
    y_hat = 1.0 / (1.0 + np.exp(-(X_train_s @ w + b)))
    w -= lr * np.mean((y_hat - y_train)[:, None] * X_train_s, axis=0)
    b -= lr * np.mean(y_hat - y_train)
prob_log = 1.0 / (1.0 + np.exp(-(X_test_s @ w + b)))

prob_knn = []
for q in X_test_s:
    d = np.sqrt(np.sum((X_train_s - q)**2, axis=1))
    tk = y_train[np.argsort(d, kind='stable')[:3]] # K=3 from part 2 output
    prob_knn.append(np.mean(tk))
prob_knn = np.array(prob_knn)

test_splits = {'S1': X_test[:, 5] == 1, 'S2': X_test[:, 6] == 1, 'S3': X_test[:, 2] > 0.5, 'S4': X_test[:, 0] <= 24}
prob_stump = np.where(test_splits[best_split_ig], stump_yes_prob, stump_no_prob)

pred_log = (prob_log >= 0.3).astype(int)
pred_knn = (prob_knn >= 0.5).astype(int)
pred_stump = (prob_stump >= 0.3).astype(int)
pred_hard = ((pred_log + pred_knn + pred_stump) >= 2).astype(int)
pred_soft = (((prob_log + prob_knn + prob_stump) / 3.0) >= 0.3).astype(int)

models = [('logistic', pred_log), ('knn', pred_knn), ('stump', pred_stump), 
          ('hard_vote', pred_hard), ('soft_vote', pred_soft)]

with open("outputs/part4_voting.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model','tp','fp','tn','fn','precision','recall','f1'])
    for name, p in models:
        tp = np.sum((p == 1) & (y_test == 1))
        fp = np.sum((p == 1) & (y_test == 0))
        tn = np.sum((p == 0) & (y_test == 0))
        fn = np.sum((p == 0) & (y_test == 1))
        prec = tp / (tp + fp) if (tp+fp)>0 else 0
        rec = tp / (tp + fn) if (tp+fn)>0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec+rec)>0 else 0
        writer.writerow([name, tp, fp, tn, fn, f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

# 4.3 AdaBoost
y_ada = np.array([1, 1, 1, -1, -1, -1, 1, -1])
h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

w1 = np.ones(8) / 8.0
err1 = np.sum(w1[h1 != y_ada])
res['round1_error'] = err1
a1 = 0.5 * math.log((1 - err1) / err1)
res['round1_alpha'] = a1

wu1 = w1 * np.exp(-a1 * y_ada * h1)
w2 = wu1 / np.sum(wu1)

err2 = np.sum(w2[h2 != y_ada])
res['round2_error'] = err2
a2 = 0.5 * math.log((1 - err2) / err2)
res['round2_alpha'] = a2
wu2 = w2 * np.exp(-a2 * y_ada * h2)
w3 = wu2 / np.sum(wu2)

ens_pred = np.sign(a1 * h1 + a2 * h2)
res['ensemble_accuracy'] = np.mean(ens_pred == y_ada)

with open("outputs/part4_adaboost.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['round','point','weight_before','weight_unnormalised','weight_after'])
    for i in range(8): writer.writerow([1, i+1, f"{w1[i]:.4f}", f"{wu1[i]:.4f}", f"{w2[i]:.4f}"])
    for i in range(8): writer.writerow([2, i+1, f"{w2[i]:.4f}", f"{wu2[i]:.4f}", f"{w3[i]:.4f}"])

# 4.4 OOB
res['oob_fraction'] = (1 - 1/560)**560

# 4.5 NN sizing
res['fc_layer_params'] = (128 * 128 * 3) * 1024 + 1024
res['conv_layer_params'] = (3 * 3 * 3) * 64 + 64
res['rnn_total'] = (128 * 64) + (128 * 128) + (16 * 128) + 128 + 16

mlp = {'mlp_hidden1': 784*16+16, 'mlp_hidden2': 16*16+16, 'mlp_output': 16*10+10}
cnn = {'cnn_conv1': 5*5*1*6+6, 'cnn_conv2': 5*5*6*16+16, 'cnn_dense1': 400*32+32, 'cnn_dense2': 32*10+10}

res['mlp_total'] = sum(mlp.values())
res['cnn_total'] = sum(cnn.values())

with open("outputs/part4_architecture.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['layer','parameters'])
    for k, v in mlp.items(): writer.writerow([k, v])
    for k, v in cnn.items(): writer.writerow([k, v])

with open("outputs/part4_results.txt", "w") as f:
    for k, v in res.items():
        if isinstance(v, float): f.write(f"{k} = {v:.4f}\n")
        else: f.write(f"{k} = {v}\n")