import csv, os, math
import numpy as np

os.makedirs("outputs", exist_ok=True)

with open("data/records.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    data = [row for row in reader]

n = len(data)
X1 = np.array([float(r[1]) for r in data])
X2 = np.array([float(r[2]) for r in data])
X3 = np.array([float(r[3]) for r in data])
X4 = np.array([float(r[4]) for r in data])
X5 = np.array([float(r[5]) for r in data])
X6 = np.array([float(r[6]) for r in data])
X7 = np.array([float(r[7]) for r in data])
outcome = np.array([float(r[8]) for r in data])
value = np.array([float(r[9]) for r in data])

res = {}

# 1.1 Estimate with an interval
res['n'] = n
res['mean_value'] = np.mean(value)
res['sd_value'] = np.std(value, ddof=1)
res['se_value'] = res['sd_value'] / math.sqrt(n)
res['ci95_low'] = res['mean_value'] - 1.96 * res['se_value']
res['ci95_high'] = res['mean_value'] + 1.96 * res['se_value']

# 1.2 Test planning claim
res['z_stat'] = (res['mean_value'] - 8.0) / res['se_value']
res['reject_h0'] = 'yes' if abs(res['z_stat']) > 1.96 else 'no'

# 1.3 Size a future sample
E = 0.5
res['n_required'] = math.ceil((1.96 * res['sd_value'] / E)**2)

# 1.4 Simple regression (closed form)
var_x4 = np.var(X4, ddof=1)
cov_x4_val = np.cov(X4, value, ddof=1)[0, 1]
res['reg_slope'] = cov_x4_val / var_x4
res['reg_intercept'] = res['mean_value'] - res['reg_slope'] * np.mean(X4)

fitted = res['reg_slope'] * X4 + res['reg_intercept']
res['tss'] = np.sum((value - res['mean_value'])**2)
res['ess'] = np.sum((fitted - res['mean_value'])**2)
res['rss'] = np.sum((value - fitted)**2)
res['tss_equals_ess_plus_rss'] = 'yes' if abs(res['tss'] - res['ess'] - res['rss']) < 1e-6 * res['tss'] else 'no'
res['r2'] = 1 - (res['rss'] / res['tss'])
res['rmse'] = math.sqrt(res['rss'] / n)
res['mae'] = np.mean(np.abs(value - fitted))

# 1.5 Gradient descent
x4_mean = np.mean(X4)
x4_sd = np.std(X4, ddof=1)
z = (X4 - x4_mean) / x4_sd
m, c, eta = 0.0, 0.0, 0.1

def cost_J(m, c): return np.mean((value - (m * z + c))**2)

res['gd_J0'] = cost_J(m, c)
for i in range(1, 501):
    e = value - (m * z + c)
    m -= eta * (-2.0 * np.mean(z * e))
    c -= eta * (-2.0 * np.mean(e))
    if i in [1, 10, 100, 500]:
        res[f'gd_J{i}'] = cost_J(m, c)

res['gd_slope'] = m / x4_sd
res['gd_intercept'] = c - m * x4_mean / x4_sd
res['gd_slope_gap'] = abs(res['gd_slope'] - res['reg_slope'])

# 1.6 Funnel check and log remedy
sorted_idx = np.argsort(fitted, kind='stable')
sorted_resids = (value - fitted)[sorted_idx]
res['resid_sd_lower_raw'] = np.std(sorted_resids[:400], ddof=1)
res['resid_sd_upper_raw'] = np.std(sorted_resids[400:], ddof=1)
res['resid_sd_ratio_raw'] = res['resid_sd_upper_raw'] / res['resid_sd_lower_raw']

log_val = np.log(value)
log_x4 = np.log(X4)
res['loglog_slope'] = np.cov(log_x4, log_val, ddof=1)[0, 1] / np.var(log_x4, ddof=1)
res['loglog_intercept'] = np.mean(log_val) - res['loglog_slope'] * np.mean(log_x4)

log_fitted = res['loglog_slope'] * log_x4 + res['loglog_intercept']
log_rss = np.sum((log_val - log_fitted)**2)
log_tss = np.sum((log_val - np.mean(log_val))**2)
res['loglog_r2'] = 1 - (log_rss / log_tss)
res['loglog_rmse'] = math.sqrt(log_rss / n)

sorted_log_resids = (log_val - log_fitted)[np.argsort(log_fitted, kind='stable')]
res['resid_sd_lower_log'] = np.std(sorted_log_resids[:400], ddof=1)
res['resid_sd_upper_log'] = np.std(sorted_log_resids[400:], ddof=1)
res['resid_sd_ratio_log'] = res['resid_sd_upper_log'] / res['resid_sd_lower_log']

# 1.7 Multiple regression
X_mat = np.column_stack([np.ones(n), log_x4, X1, X2, X3, X5, X6, X7])
beta = np.linalg.solve(X_mat.T @ X_mat, X_mat.T @ log_val)

for i in range(8):
    res[f'mr_beta_{i}'] = beta[i]

mr_r2 = 1 - (np.sum((log_val - (X_mat @ beta))**2) / log_tss)
res['mr_r2'] = mr_r2
res['mr_adj_r2'] = 1 - (1 - mr_r2) * (n - 1) / (n - 7 - 1)
res['loglog_adj_r2'] = 1 - (1 - res['loglog_r2']) * (n - 1) / (n - 1 - 1)

with open("outputs/part1_results.txt", "w") as f:
    for k, v in res.items():
        if isinstance(v, float): f.write(f"{k} = {v:.4f}\n")
        else: f.write(f"{k} = {v}\n")