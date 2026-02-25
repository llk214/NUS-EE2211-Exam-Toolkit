import numpy as np
import re
import io

from gui.utils import parse_matrix


def gini(y):
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return 1 - np.sum(p * p)


def entropy_fn(y):
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def mse_impurity(y):
    if len(y) == 0:
        return 0.0
    return np.var(y)


def parse_thresholds(s, n_features):
    if not s or s.strip() == "":
        return None
    s = s.strip()
    thr_map = {}
    parts = [p.strip() for p in re.split('[;,]', s) if p.strip()]
    for p in parts:
        if ':' in p:
            fi, vals = p.split(':', 1)
            fi = int(fi.strip())
            nums = [float(x) for x in vals.split() if x.strip()]
            thr_map[fi] = nums
        else:
            nums = [float(x) for x in p.split() if x.strip()]
            for fi in range(n_features):
                thr_map.setdefault(fi, nums)
    thr_map = {k: v for k, v in thr_map.items() if 0 <= k < n_features and len(v) > 0}
    return thr_map if thr_map else None


def best_split_classification(X, y, crit, feat_idx, thr_map=None):
    imp_fn = gini if crit == "gini" else entropy_fn
    parent = imp_fn(y)
    n = len(y)
    best = None
    best_thr = None
    best_gain = -1
    best_child = None

    for j in feat_idx:
        vals = np.unique(X[:, j])
        if len(vals) <= 1:
            continue
        if thr_map is not None and j in thr_map:
            thrs = np.array(sorted(thr_map[j]))
        else:
            thrs = (vals[:-1] + vals[1:]) / 2
        for thr in thrs:
            L = X[:, j] <= thr
            R = ~L
            if L.sum() == 0 or R.sum() == 0:
                continue
            impL = imp_fn(y[L])
            impR = imp_fn(y[R])
            child = (L.sum() * impL + R.sum() * impR) / n
            gain_val = parent - child
            if gain_val > best_gain:
                best_gain = gain_val
                best = j
                best_thr = thr
                best_child = child
    return best, best_thr, best_gain, parent, best_child


def best_split_regression(X, y, feat_idx, thr_map=None):
    parent_mse = mse_impurity(y)
    n = len(y)
    best = None
    best_thr = None
    best_gain = -1
    best_child_mse = None

    for j in feat_idx:
        vals = np.unique(X[:, j])
        if len(vals) <= 1:
            continue
        if thr_map is not None and j in thr_map:
            thrs = np.array(sorted(thr_map[j]))
        else:
            thrs = (vals[:-1] + vals[1:]) / 2
        for thr in thrs:
            L = X[:, j] <= thr
            R = ~L
            if L.sum() == 0 or R.sum() == 0:
                continue
            mse_L = mse_impurity(y[L])
            mse_R = mse_impurity(y[R])
            child_mse = (L.sum() * mse_L + R.sum() * mse_R) / n
            gain_val = parent_mse - child_mse
            if gain_val > best_gain:
                best_gain = gain_val
                best = j
                best_thr = thr
                best_child_mse = child_mse
    return best, best_thr, best_gain, parent_mse, best_child_mse


class TreeNode:
    def __init__(self, f=None, t=None, l=None, r=None, pred=None):
        self.f, self.t, self.l, self.r, self.pred = f, t, l, r, pred

    def leaf(self):
        return self.pred is not None


def majority(y):
    v, c = np.unique(y, return_counts=True)
    return v[np.argmax(c)]


def build_cls_tree(X, y, d, maxd, mins, crit, feat_sub, thr_map=None):
    if d >= maxd or len(np.unique(y)) == 1 or len(y) < mins:
        return TreeNode(pred=majority(y))
    nfeat = X.shape[1]
    idx = np.random.choice(nfeat, min(feat_sub, nfeat), replace=False) if feat_sub < nfeat else np.arange(nfeat)
    f, thr, g, parent, ch = best_split_classification(X, y, crit, idx, thr_map)
    if f is None:
        return TreeNode(pred=majority(y))
    L = X[:, f] <= thr
    return TreeNode(f, thr,
                    build_cls_tree(X[L], y[L], d + 1, maxd, mins, crit, feat_sub, thr_map),
                    build_cls_tree(X[~L], y[~L], d + 1, maxd, mins, crit, feat_sub, thr_map))


def build_reg_tree(X, y, d, maxd, mins, feat_sub, thr_map=None):
    if d >= maxd or len(y) < mins or np.var(y) < 1e-10:
        return TreeNode(pred=np.mean(y))
    nfeat = X.shape[1]
    idx = np.random.choice(nfeat, min(feat_sub, nfeat), replace=False) if feat_sub < nfeat else np.arange(nfeat)
    f, thr, g, p_mse, c_mse = best_split_regression(X, y, idx, thr_map)
    if f is None:
        return TreeNode(pred=np.mean(y))
    L = X[:, f] <= thr
    return TreeNode(f, thr,
                    build_reg_tree(X[L], y[L], d + 1, maxd, mins, feat_sub, thr_map),
                    build_reg_tree(X[~L], y[~L], d + 1, maxd, mins, feat_sub, thr_map))


def predict_tree(root, X):
    out = []
    for x in X:
        n = root
        while not n.leaf():
            n = n.l if x[n.f] <= n.t else n.r
        out.append(n.pred)
    return np.array(out)


def compute_tree(X_str, y_str, task, criterion, thr_str, tree_mode, depth, min_samples,
                 n_trees, feat_mode):
    out = io.StringIO()
    result = {'text': '', 'weights': None, 'bias': None, 'iterations': [],
              'tree': None, 'trees': None, 'predictions': None, 'metrics': None}
    X = parse_matrix(X_str)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = parse_matrix(y_str).flatten()
    if task == 'classification':
        y = y.astype(int)

    thr_map = parse_thresholds(thr_str, X.shape[1])

    if tree_mode == 'root':
        if task == 'classification':
            f, thr, g, p, c = best_split_classification(X, y, criterion, np.arange(X.shape[1]), thr_map)
            out.write(f"Parent Impurity ({criterion}): {p:.8f}\n")
            if f is not None:
                out.write(f"Best Split: Feature index {f}, Threshold {thr:.8f}\n")
                out.write(f"Gain: {g:.8f}\n")
                out.write(f"Child Impurity: {c:.8f}\n")
            else:
                out.write("No split possible.\n")
        else:
            f, thr, g, p_mse, c_mse = best_split_regression(X, y, np.arange(X.shape[1]), thr_map)
            out.write(f"Parent MSE: {p_mse:.8f}\n")
            if f is not None:
                out.write(f"Best Split: Feature index {f}, Threshold {thr:.8f}\n")
                out.write(f"MSE Reduction: {g:.8f}\n")
                out.write(f"Child MSE: {c_mse:.8f}\n")
            else:
                out.write("No split possible.\n")

    elif tree_mode == 'tree':
        sub = X.shape[1]
        if task == 'classification':
            tree = build_cls_tree(X, y, 0, depth, min_samples, criterion, sub, thr_map)
        else:
            tree = build_reg_tree(X, y, 0, depth, min_samples, sub, thr_map)
        out.write("Tree built.\n")
        train_pred = predict_tree(tree, X)
        if task == 'classification':
            acc = np.mean(train_pred == y)
            out.write(f"Training Accuracy: {acc:.4f}\n")
        else:
            mse_val = np.mean((train_pred - y) ** 2)
            out.write(f"Training MSE: {mse_val:.8f}\n")
            out.write(f"Training RMSE: {np.sqrt(mse_val):.8f}\n")
        out.write(f"Predictions: {np.round(train_pred, 8)}\n")
        result['tree'] = tree
        result['predictions'] = train_pred

    elif tree_mode == 'forest':
        fm_val = feat_mode
        n = X.shape[0]
        k = X.shape[1]
        if fm_val == "sqrt":
            sub = max(1, int(np.sqrt(k)))
        elif fm_val == "log2":
            sub = max(1, int(np.log2(k)))
        else:
            sub = k

        trees = []
        for _ in range(n_trees):
            idx = np.random.randint(0, n, n)
            Xb, yb = X[idx], y[idx]
            if task == 'classification':
                t = build_cls_tree(Xb, yb, 0, depth, min_samples, criterion, sub, thr_map)
            else:
                t = build_reg_tree(Xb, yb, 0, depth, min_samples, sub, thr_map)
            trees.append(t)

        out.write(f"Forest built ({n_trees} trees).\n")
        allp = np.array([predict_tree(t, X) for t in trees])
        if task == 'classification':
            preds = []
            for j in range(X.shape[0]):
                v, c = np.unique(allp[:, j], return_counts=True)
                preds.append(v[np.argmax(c)])
            preds = np.array(preds)
            acc = np.mean(preds == y)
            out.write(f"Training Accuracy: {acc:.4f}\n")
        else:
            preds = np.mean(allp, axis=0)
            mse_val = np.mean((preds - y) ** 2)
            out.write(f"Training MSE: {mse_val:.8f}\n")
        out.write(f"Predictions: {np.round(preds, 8)}\n")
        result['trees'] = trees
        result['predictions'] = preds

    result['text'] = out.getvalue()
    result['summary_text'] = out.getvalue()
    return result
