"""
EE2211 Exam Toolkit - GUI Version
Tkinter-based GUI for all EE2211 ML modules.
Zero external dependencies beyond numpy.
"""

import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import tkinter as tk
from tkinter import ttk
import numpy as np
from itertools import combinations_with_replacement
import math
import re
import io


# ============================================================================
# UTILITY: Parse helpers
# ============================================================================

def parse_matrix(s):
    """Parse comma-separated rows into a numpy matrix.
    '1 2 3, 4 5 6' -> [[1,2,3],[4,5,6]]
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty input")
    rows = [r.strip() for r in s.split(',') if r.strip()]
    return np.array([list(map(float, r.split())) for r in rows], dtype=float)


def parse_vector(s):
    """Parse space-separated values into a 1D array."""
    s = s.strip()
    if not s:
        raise ValueError("Empty input")
    return np.array(list(map(float, s.split())), dtype=float)


# ============================================================================
# COMPUTATION: Classification
# ============================================================================

def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-z))


def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def bce_loss(y_true, p):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def cce_loss(Y_true, P):
    eps = 1e-12
    P = np.clip(P, eps, 1 - eps)
    return -np.mean(np.sum(Y_true * np.log(P), axis=1))


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def parse_y_classification(y_str):
    if ',' in y_str:
        Y = parse_matrix(y_str)
        K = Y.shape[1]
        if K == 2:
            y = np.argmax(Y, axis=1).astype(int)
            return 'binary', y, 2
        return 'multiclass', Y, K
    else:
        y = np.array(list(map(float, y_str.split())), dtype=float).astype(int)
        classes = np.unique(y)
        if set(classes.tolist()) <= {0, 1}:
            return 'binary', y, 2
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([class_to_idx[c] for c in y], dtype=int)
        K = len(classes)
        Y = np.zeros((len(y), K), dtype=float)
        Y[np.arange(len(y)), y_idx] = 1
        return 'multiclass', Y, K


def compute_classification(X_str, y_str, mode_choice, lr, iters, batch_type, batch_size,
                           momentum, l2, penalize_bias, threshold, w_init_choice, w_init_str,
                           pred_binary, pred_W_str, pred_threshold, show_metrics):
    out = io.StringIO()
    result = {'text': '', 'weights': None, 'bias': None, 'iterations': [],
              'tree': None, 'trees': None, 'predictions': None, 'metrics': None}

    X = parse_matrix(X_str)

    if mode_choice == 'predict':
        if pred_binary:
            W = parse_matrix(pred_W_str).flatten()
            Xb = add_bias(X)
            p = sigmoid(Xb @ W)
            th = pred_threshold
            preds = (p >= th).astype(int)
            out.write(f"Probabilities: {np.round(p, 8)}\n")
            out.write(f"Labels: {preds}\n")
            result['predictions'] = preds
        else:
            W = parse_matrix(pred_W_str)
            Xb = add_bias(X)
            P = softmax(Xb @ W)
            preds = np.argmax(P, axis=1)
            out.write(f"Probabilities:\n{np.round(P, 8)}\n")
            out.write(f"Labels: {preds}\n")
            result['predictions'] = preds
        result['text'] = out.getvalue()
        return result

    mode, y_or_Y, K = parse_y_classification(y_str)

    config = dict(lr=lr, iters=iters, batch=batch_type, batch_size=batch_size,
                  momentum=momentum, l2=l2, penalize_bias=penalize_bias,
                  print_every=max(1, iters // 20), tol=1e-9, threshold=threshold)

    summary_out = io.StringIO()

    if mode == 'binary':
        Xb = add_bias(X)
        n, d1 = Xb.shape
        if w_init_choice == 'manual':
            W = parse_matrix(w_init_str).flatten().astype(float)
        elif w_init_choice == 'random':
            W = np.random.randn(d1)
        else:
            W = np.zeros(d1)

        opt_state = {'v': np.zeros_like(W)}
        bs = batch_size if batch_size > 0 else n
        loss_history = []

        for epoch in range(1, iters + 1):
            idxs = np.arange(n)
            if batch_type != 'gd':
                np.random.shuffle(idxs)
            for start in range(0, n, bs):
                batch_idx = idxs[start:start + bs]
                Xb_batch = Xb[batch_idx]
                y_batch = y_or_Y[batch_idx]
                p = sigmoid(Xb_batch @ W)
                grad = (Xb_batch.T @ (p - y_batch)) / len(y_batch)
                grad = grad.reshape(-1)
                if l2 != 0:
                    W_reg = W.copy()
                    if not penalize_bias:
                        W_reg[0] = 0.0
                    grad = grad + l2 * W_reg
                v = opt_state.get('v', np.zeros_like(W))
                v = momentum * v + lr * grad
                opt_state['v'] = v
                W = W - v

            p_all = sigmoid(Xb @ W)
            loss = bce_loss(y_or_Y, p_all)
            loss_history.append(loss)
            if epoch % config['print_every'] == 0 or epoch == 1 or epoch == iters:
                acc = np.mean((p_all >= threshold).astype(int) == y_or_Y)
                iter_text = f"[BIN] epoch {epoch}/{iters} | loss={loss:.8f} acc={acc:.4f}\n"
                out.write(iter_text)
                result['iterations'].append({
                    'iter': epoch, 'text': iter_text.strip(),
                    'loss': float(loss), 'W': W.copy()
                })
            if len(loss_history) > 2 and abs(loss_history[-1] - loss_history[-2]) < 1e-9:
                out.write("Early stop: loss change < tol\n")
                summary_out.write("Early stop: loss change < tol\n")
                break

        out.write(f"\nTrained W: {np.round(W, 8)}\n")
        summary_out.write(f"Trained W: {np.round(W, 8)}\n")
        result['weights'] = W.copy()

        if show_metrics:
            p = sigmoid(Xb @ W)
            yhat = (p >= threshold).astype(int)
            ytrue = y_or_Y
            TP = int(np.sum((yhat == 1) & (ytrue == 1)))
            TN = int(np.sum((yhat == 0) & (ytrue == 0)))
            FP = int(np.sum((yhat == 1) & (ytrue == 0)))
            FN = int(np.sum((yhat == 0) & (ytrue == 1)))
            prec = TP / (TP + FP) if TP + FP > 0 else 0.0
            rec = TP / (TP + FN) if TP + FN > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            acc = (TP + TN) / (TP + TN + FP + FN)
            metrics_text = f"\nConfusion: TP={TP}, FP={FP}, FN={FN}, TN={TN}\n"
            metrics_text += f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}\n"
            out.write(metrics_text)
            summary_out.write(metrics_text)
            result['metrics'] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

    else:
        Xb = add_bias(X)
        n, d1 = Xb.shape
        if w_init_choice == 'manual':
            W = parse_matrix(w_init_str).astype(float)
        elif w_init_choice == 'random':
            W = np.random.randn(d1, K)
        else:
            W = np.zeros((d1, K))

        opt_state = {'v': np.zeros_like(W)}
        bs = batch_size if batch_size > 0 else n
        loss_history = []

        for epoch in range(1, iters + 1):
            idxs = np.arange(n)
            if batch_type != 'gd':
                np.random.shuffle(idxs)
            for start in range(0, n, bs):
                batch_idx = idxs[start:start + bs]
                Xb_batch = Xb[batch_idx]
                Y_batch = y_or_Y[batch_idx]
                P = softmax(Xb_batch @ W)
                grad = (Xb_batch.T @ (P - Y_batch)) / len(Y_batch)
                if l2 != 0:
                    W_reg = W.copy()
                    if not penalize_bias:
                        W_reg[0, :] = 0.0
                    grad = grad + l2 * W_reg
                v = opt_state.get('v', np.zeros_like(W))
                v = momentum * v + lr * grad
                opt_state['v'] = v
                W = W - v

            P_all = softmax(Xb @ W)
            loss = cce_loss(y_or_Y, P_all)
            loss_history.append(loss)
            if epoch % config['print_every'] == 0 or epoch == 1 or epoch == iters:
                acc = np.mean(np.argmax(P_all, axis=1) == np.argmax(y_or_Y, axis=1))
                iter_text = f"[MC ] epoch {epoch}/{iters} | loss={loss:.8f} acc={acc:.4f}\n"
                out.write(iter_text)
                result['iterations'].append({
                    'iter': epoch, 'text': iter_text.strip(),
                    'loss': float(loss), 'W': W.copy()
                })
            if len(loss_history) > 2 and abs(loss_history[-1] - loss_history[-2]) < 1e-9:
                out.write("Early stop: loss change < tol\n")
                summary_out.write("Early stop: loss change < tol\n")
                break

        out.write(f"\nTrained W:\n{np.round(W, 8)}\n")
        summary_out.write(f"Trained W:\n{np.round(W, 8)}\n")
        result['weights'] = W.copy()

        if show_metrics:
            P = softmax(Xb @ W)
            yhat = np.argmax(P, axis=1)
            ytrue = np.argmax(y_or_Y, axis=1)
            cm = np.zeros((K, K), dtype=int)
            for a, b in zip(ytrue, yhat):
                cm[a, b] += 1
            metrics_text = f"\nConfusion matrix (rows=true, cols=pred):\n{cm}\n"
            out.write(metrics_text)
            summary_out.write(metrics_text)

    result['text'] = out.getvalue()
    result['summary_text'] = summary_out.getvalue()
    return result


# ============================================================================
# COMPUTATION: Clustering
# ============================================================================

def pairwise_sq_dists(X, C):
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(C * C, axis=1, keepdims=True).T
    return x2 + c2 - 2 * (X @ C.T)


def compute_clustering(X_str, method, C0_str, K_val, fuzzifier, max_iter, tol):
    out = io.StringIO()
    result = {'text': '', 'weights': None, 'bias': None, 'iterations': [],
              'tree': None, 'trees': None, 'predictions': None, 'metrics': None}
    X = parse_matrix(X_str)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if method in ('kmeans', 'fcm'):
        C = parse_matrix(C0_str)
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        K = C.shape[0]
    else:
        K = K_val
        # K-means++ init
        rng = np.random.default_rng()
        n = X.shape[0]
        C = np.empty((K, X.shape[1]))
        idx = rng.integers(0, n)
        C[0] = X[idx]
        for k in range(1, K):
            d2 = np.min(pairwise_sq_dists(X, C[:k]), axis=1)
            probs = d2 / (np.sum(d2) + 1e-12)
            idx = rng.choice(n, p=probs)
            C[k] = X[idx]

    out.write(f"Initial Centroids:\n{np.round(C, 8)}\n")
    summary_out = io.StringIO()

    if method == 'fcm':
        centers = C.copy()
        m = fuzzifier
        for iteration in range(max_iter):
            # Update membership
            n_samples = X.shape[0]
            n_clusters = centers.shape[0]
            W = np.zeros((n_samples, n_clusters))
            for i in range(n_samples):
                for k in range(n_clusters):
                    denom = 0.0
                    dist_k = np.linalg.norm(X[i] - centers[k]) + 1e-10
                    for j in range(n_clusters):
                        dist_j = np.linalg.norm(X[i] - centers[j]) + 1e-10
                        denom += (dist_k / dist_j) ** (2 / (m - 1))
                    W[i, k] = 1 / denom

            # Update centers
            new_centers = np.zeros_like(centers)
            for k in range(n_clusters):
                numerator = X.T @ (W[:, k] ** m)
                denominator = np.sum(W[:, k] ** m)
                new_centers[k] = numerator / denominator

            # Objective
            J = 0.0
            for i in range(n_samples):
                for k in range(n_clusters):
                    dist_sq = np.sum((X[i] - new_centers[k]) ** 2)
                    J += (W[i, k] ** m) * dist_sq

            labels = np.argmax(W, axis=1)
            centroid_change = np.linalg.norm(new_centers - centers)

            iter_text = f"--- FCM Iteration {iteration} ---\n"
            iter_text += f"Centroids:\n{np.round(new_centers, 8)}\n"
            iter_text += f"Membership W:\n{np.round(W, 8)}\n"
            iter_text += f"Labels (hard): {labels}\n"
            iter_text += f"Objective J = {J:.8f}\n"
            iter_text += f"Centroid change = {centroid_change:.8e}\n"
            out.write(f"\n{iter_text}")
            result['iterations'].append({
                'iter': iteration, 'text': iter_text.strip(),
                'loss': float(J), 'W': new_centers.copy()
            })

            if centroid_change < tol:
                out.write(f"Converged at iteration {iteration}\n")
                centers = new_centers
                break
            centers = new_centers

        out.write(f"\nFinal Centroids:\n{np.round(centers, 8)}\n")
        out.write(f"Final Labels: {np.argmax(W, axis=1)}\n")
        summary_out.write(f"Final Centroids:\n{np.round(centers, 8)}\n")
        summary_out.write(f"Final Labels: {np.argmax(W, axis=1)}\n")
        result['weights'] = centers.copy()

    else:
        # K-means
        last_J = None
        for t in range(max_iter):
            D2 = pairwise_sq_dists(X, C)
            labels = np.argmin(D2, axis=1)
            C_new = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k)
                else X[np.random.randint(0, X.shape[0])]
                for k in range(K)
            ])
            min_dists = np.min(pairwise_sq_dists(X, C_new), axis=1)
            J = float(np.sum(min_dists))

            iter_text = f"--- K-means Iteration {t} ---\n"
            iter_text += f"Centroids:\n{np.round(C_new, 8)}\n"
            iter_text += f"Labels: {labels}\n"
            iter_text += f"Distortion J={J:.8f}\n"
            out.write(f"\n{iter_text}")
            result['iterations'].append({
                'iter': t, 'text': iter_text.strip(),
                'loss': float(J), 'W': C_new.copy()
            })

            if last_J is not None and abs(last_J - J) < 1e-6:
                out.write("Converged: dJ < 1e-6\n")
                C = C_new
                break
            C, last_J = C_new, J

        out.write(f"\nFinal distortion J={J:.8f}, converged at iter={t}\n")
        summary_out.write(f"Final distortion J={J:.8f}, converged at iter={t}\n")
        summary_out.write(f"Final Centroids:\n{np.round(C, 8)}\n")
        result['weights'] = C.copy()

    result['text'] = out.getvalue()
    result['summary_text'] = summary_out.getvalue()
    return result


# ============================================================================
# COMPUTATION: Gradient Descent (Linear / Softmax)
# ============================================================================

def compute_gradient_descent(mode, X_str, y_str, W0_str, lr, iters, add_bias_col, normalize,
                             n_classes, labels_1indexed):
    out = io.StringIO()
    result = {'text': '', 'weights': None, 'bias': None, 'iterations': [],
              'tree': None, 'trees': None, 'predictions': None, 'metrics': None}
    X = parse_matrix(X_str)
    if add_bias_col:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        out.write(f"X with bias:\n{X}\n\n")

    summary_out = io.StringIO()

    if mode == 'softmax':
        y_raw = list(map(int, parse_matrix(y_str).flatten()))
        if labels_1indexed:
            y = np.array(y_raw) - 1
            out.write(f"Converted to 0-indexed: {y}\n")
        else:
            y = np.array(y_raw)

        W = parse_matrix(W0_str).astype(float)
        n_samples = X.shape[0]
        Y_onehot = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            Y_onehot[i, y[i]] = 1

        out.write(f"Y one-hot:\n{Y_onehot.astype(int)}\n")
        out.write(f"Initial W:\n{np.round(W, 8)}\n")

        for t in range(1, iters + 1):
            Z = X @ W
            P = softmax(Z)
            gradient = X.T @ (P - Y_onehot)
            if normalize:
                gradient /= n_samples
            W = W - lr * gradient

            loss = -np.sum(Y_onehot * np.log(P + 1e-10))
            if normalize:
                loss /= n_samples
            labels = np.argmax(P, axis=1)
            accuracy = np.mean(labels == y)

            iter_text = f"--- Iteration {t} ---\n"
            iter_text += f"Z (logits):\n{np.round(Z, 8)}\n"
            iter_text += f"P (softmax):\n{np.round(P, 8)}\n"
            iter_text += f"Gradient:\n{np.round(gradient, 8)}\n"
            iter_text += f"W:\n{np.round(W, 8)}\n"
            iter_text += f"Loss = {loss:.8f}, Accuracy = {accuracy:.4f}\n"
            iter_text += f"Predicted labels: {labels}\n"
            out.write(f"\n{iter_text}")
            result['iterations'].append({
                'iter': t, 'text': iter_text.strip(),
                'loss': float(loss), 'W': W.copy()
            })

        out.write(f"\nFinal W:\n{np.round(W, 8)}\n")
        summary_out.write(f"Final W:\n{np.round(W, 8)}\n")
        result['weights'] = W.copy()

    else:
        y = parse_matrix(y_str).flatten()
        w = parse_matrix(W0_str).flatten()
        m, n = X.shape
        factor = (2.0 / m) if normalize else 2.0

        for t in range(1, iters + 1):
            y_hat = X @ w
            grad = factor * (X.T @ (y_hat - y))
            w = w - lr * grad
            cost = np.sum((y_hat - y) ** 2)
            if normalize:
                cost /= m
            iter_text = f"iter {t}: w = {np.round(w, 8)}, cost = {cost:.8f}\n"
            out.write(iter_text)
            result['iterations'].append({
                'iter': t, 'text': iter_text.strip(),
                'loss': float(cost), 'W': w.copy()
            })

        out.write(f"\nFinal w: {np.round(w, 8)}\n")
        summary_out.write(f"Final w: {np.round(w, 8)}\n")
        result['weights'] = w.copy()

    result['text'] = out.getvalue()
    result['summary_text'] = summary_out.getvalue()
    return result


# ============================================================================
# COMPUTATION: Neural Network
# ============================================================================

def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)
def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

NN_ACT_FUNCS = {
    "relu": (relu, relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "linear": (lambda z: z, lambda z: np.ones_like(z)),
    "softmax": (softmax, None),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
}


def nn_forward(X, weights, activations):
    caches = []
    A = X
    for i, (W, act_name) in enumerate(zip(weights, activations)):
        act, _ = NN_ACT_FUNCS[act_name]
        A_b = np.concatenate([np.ones((A.shape[0], 1)), A], axis=1)
        Z = A_b @ W
        if act_name == "softmax":
            A = softmax(Z)
        else:
            A = act(Z)
        caches.append((A_b, Z, act_name))
    return A, caches


def nn_backward(X, Y, Yhat, weights, caches, activations, loss_type="mse"):
    grads = []
    N = X.shape[0]
    if loss_type == "mse":
        dA = (2.0 / N) * (Yhat - Y)
    else:
        dA = (Yhat - Y) / N

    for i in reversed(range(len(weights))):
        A_b, Z, act_name = caches[i]
        if act_name == "softmax" and loss_type != "mse":
            dZ = dA
        else:
            _, act_grad = NN_ACT_FUNCS[act_name]
            dZ = dA * act_grad(Z)
        gW = A_b.T @ dZ
        grads.insert(0, gW)
        if i > 0:
            dA_b = dZ @ weights[i].T
            dA = dA_b[:, 1:]
    return grads


def nn_init_weights(in_dim, out_dim, init_mode, seed=None):
    rows = in_dim + 1
    cols = out_dim
    if seed is not None and seed != "":
        np.random.seed(int(seed))
    if init_mode == "zeros":
        return np.zeros((rows, cols))
    if init_mode == "xavier":
        std = np.sqrt(2.0 / (in_dim + out_dim))
        return np.random.randn(rows, cols) * std
    if init_mode == "he":
        std = np.sqrt(2.0 / max(1, in_dim))
        return np.random.randn(rows, cols) * std
    if init_mode == "random":
        return np.random.randn(rows, cols)
    return np.zeros((rows, cols))


def compute_neural(X_str, Y_str, layer_configs, lr, iters, loss_type):
    """
    layer_configs: list of dicts with keys:
      'neurons': int, 'activation': str, 'init': str, 'seed': str, 'W_manual': str or None
    """
    out = io.StringIO()
    result = {'text': '', 'weights': None, 'bias': None, 'iterations': [],
              'tree': None, 'trees': None, 'predictions': None, 'metrics': None}
    X = parse_matrix(X_str)
    Y = parse_matrix(Y_str)

    activations = []
    weights = []
    prev_dim = X.shape[1]

    for i, lc in enumerate(layer_configs):
        act = lc['activation']
        out_dim = lc['neurons']
        activations.append(act)

        if lc['init'] == 'manual' and lc.get('W_manual'):
            W = parse_matrix(lc['W_manual'])
            if W.shape != (prev_dim + 1, out_dim):
                flat = W.flatten()
                expected = (prev_dim + 1) * out_dim
                if flat.size == expected:
                    W = flat.reshape(prev_dim + 1, out_dim)
                else:
                    out.write(f"  Layer {i+1}: W shape mismatch, falling back to xavier\n")
                    W = nn_init_weights(prev_dim, out_dim, "xavier")
        else:
            W = nn_init_weights(prev_dim, out_dim, lc['init'], seed=lc.get('seed'))

        out.write(f"Layer {i+1}: {prev_dim}->{out_dim}, act={act}, W shape={W.shape}\n")
        weights.append(W)
        prev_dim = out_dim

    out.write(f"\nTraining with lr={lr}, iters={iters}, loss={loss_type}\n")

    summary_out = io.StringIO()

    for t in range(1, iters + 1):
        Yhat, caches = nn_forward(X, weights, activations)
        if loss_type == "mse":
            loss = np.mean(np.sum((Yhat - Y) ** 2, axis=1))
        else:
            loss = -np.mean(np.sum(Y * np.log(np.clip(Yhat, 1e-12, 1 - 1e-12)), axis=1))
        grads = nn_backward(X, Y, Yhat, weights, caches, activations, loss_type=loss_type)
        weights = [W - lr * g for W, g in zip(weights, grads)]

        iter_text = f"Iter {t}: loss = {loss:.8f}\n"
        iter_text += f"Yhat:\n{np.round(Yhat, 8)}\n"
        out.write(f"\n{iter_text}")
        result['iterations'].append({
            'iter': t, 'text': iter_text.strip(),
            'loss': float(loss), 'W': [w.copy() for w in weights]
        })

    for i, W in enumerate(weights, 1):
        w_text = f"\nUpdated W^{i} (shape {W.shape}):\n{np.round(W, 8)}\n"
        out.write(w_text)
        summary_out.write(w_text)

    result['text'] = out.getvalue()
    result['summary_text'] = summary_out.getvalue()
    result['weights'] = [w.copy() for w in weights]
    return result


# ============================================================================
# COMPUTATION: Regression
# ============================================================================

def _as_2d(a):
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _pearson_r(Y, Y_pred):
    Y = Y.flatten()
    Y_pred = Y_pred.flatten()
    mean_y = np.mean(Y)
    mean_pred = np.mean(Y_pred)
    numerator = np.sum((Y - mean_y) * (Y_pred - mean_pred))
    denominator = np.sqrt(np.sum((Y - mean_y) ** 2) * np.sum((Y_pred - mean_pred) ** 2))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _metrics(Y, Y_pred, p):
    residuals = Y - Y_pred
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((Y - np.mean(Y, axis=0)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    n = Y.shape[0]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else float('nan')
    r = _pearson_r(Y, Y_pred)
    return mse, rmse, mae, r2, adj_r2, r


class PolynomialFeatures:
    def __init__(self, degree):
        self.degree = degree
        self.combos = None

    def fit(self, X):
        X = _as_2d(X)
        self.input_dim = X.shape[1]
        self.combos = []
        for d in range(1, self.degree + 1):
            self.combos.extend(combinations_with_replacement(range(self.input_dim), d))
        return self

    def transform(self, X):
        X = _as_2d(X)
        if self.combos is None:
            self.fit(X)
        n = X.shape[0]
        out_cols = []
        for combo in self.combos:
            col = np.ones(n)
            for idx in combo:
                col *= X[:, idx]
            out_cols.append(col.reshape(-1, 1))
        return np.hstack(out_cols) if out_cols else np.empty((n, 0))


def compute_regression(X_str, Y_str, model, alpha, degree, penalize_bias):
    out = io.StringIO()
    result = {'text': '', 'weights': None, 'bias': None, 'iterations': [],
              'tree': None, 'trees': None, 'predictions': None, 'metrics': None}
    X = _as_2d(parse_matrix(X_str))
    Y = _as_2d(parse_matrix(Y_str))

    poly = None

    if model == 'ols':
        n, p = X.shape
        Xa = np.hstack([X, np.ones((n, 1))])
        Theta, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
        W, b = Theta[:-1, :], Theta[-1, :]
        Yp = Xa @ Theta
        mets = _metrics(Y, Yp, p=p)
        out.write("=== OLS ===\n")

    elif model == 'ridge':
        n, p = X.shape
        Xa = np.hstack([X, np.ones((n, 1))])
        Reg = np.eye(p + 1) * alpha
        if not penalize_bias:
            Reg[-1, -1] = 0.0
        A = Xa.T @ Xa + Reg
        B = Xa.T @ Y
        try:
            Theta = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            Theta = np.linalg.pinv(A) @ B
        W, b = Theta[:-1, :], Theta[-1, :]
        Yp = Xa @ Theta
        mets = _metrics(Y, Yp, p=p)
        out.write(f"=== Ridge (alpha={alpha}, pen_bias={penalize_bias}) ===\n")

    elif model == 'polynomial':
        poly = PolynomialFeatures(degree).fit(X)
        Phi = poly.transform(X)
        n, p = Phi.shape
        Xa = np.hstack([Phi, np.ones((n, 1))])
        if alpha > 0:
            Reg = np.eye(p + 1) * alpha
            if not penalize_bias:
                Reg[-1, -1] = 0.0
            A = Xa.T @ Xa + Reg
            B = Xa.T @ Y
            try:
                Theta = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                Theta = np.linalg.pinv(A) @ B
        else:
            Theta, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
        W, b = Theta[:-1, :], Theta[-1, :]
        Yp = Xa @ Theta
        mets = _metrics(Y, Yp, p=p)
        out.write(f"=== Polynomial (degree={degree}, alpha={alpha}) ===\n")

    out.write(f"Model: Y = XW + b\n")
    out.write(f"W (coeffs) =\n{np.round(W, 8)}\n")
    out.write(f"b (intercept) = {np.round(b, 8)}\n\n")

    mse, rmse, mae, r2, adj_r2, r = mets
    out.write(f"MSE:         {mse:.8f}\n")
    out.write(f"RMSE:        {rmse:.8f}\n")
    out.write(f"MAE:         {mae:.8f}\n")
    out.write(f"R2:          {r2:.8f}\n")
    out.write(f"Adjusted R2: {adj_r2:.8f}\n" if not np.isnan(adj_r2) else "Adjusted R2: N/A\n")
    out.write(f"Pearson r:   {r:.8f}\n")

    result['text'] = out.getvalue()
    result['summary_text'] = out.getvalue()
    result['weights'] = W.copy()
    result['bias'] = b.copy()
    result['metrics'] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'adj_r2': adj_r2, 'r': r}
    return result


def compute_regression_predict(X_str, W_str, b_str, model, degree):
    """Predict Y = X @ W + b using given weights."""
    out = io.StringIO()
    result = {'text': '', 'weights': None, 'bias': None, 'iterations': [],
              'tree': None, 'trees': None, 'predictions': None, 'metrics': None}
    X = _as_2d(parse_matrix(X_str))
    W = _as_2d(parse_matrix(W_str))
    b = parse_matrix(b_str).flatten()

    if model == 'polynomial':
        poly = PolynomialFeatures(degree).fit(X)
        X = poly.transform(X)
        out.write(f"Polynomial features (degree={degree}):\n{np.round(X, 8)}\n\n")

    Yp = X @ W + b
    out.write(f"=== Predict ({model}) ===\n")
    out.write(f"Y_pred = X @ W + b\n")
    out.write(f"Y_pred =\n{np.round(Yp, 8)}\n")
    result['text'] = out.getvalue()
    result['predictions'] = Yp
    return result


# ============================================================================
# COMPUTATION: Decision Tree
# ============================================================================

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


# ============================================================================
# COMPUTATION: Cost Function Minimizer
# ============================================================================

MATH_NS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "abs": abs, "pi": math.pi, "e": math.e,
    "ln": math.log,
}


def compute_cost_minimizer(mode, expr, var_names, init_vals, lr, iters):
    out = io.StringIO()
    result = {'text': '', 'weights': None, 'bias': None, 'iterations': [],
              'tree': None, 'trees': None, 'predictions': None, 'metrics': None}
    h = 1e-6
    vals = {name: float(val) for name, val in zip(var_names, init_vals)}
    ns_base = {"__builtins__": {}, "math": math, "np": np, **MATH_NS}

    for t in range(1, iters + 1):
        C = eval(expr, {**ns_base, **vals})
        grads = {}
        for name in var_names:
            orig = vals[name]
            vals[name] = orig + h
            Cp = eval(expr, {**ns_base, **vals})
            vals[name] = orig - h
            Cm = eval(expr, {**ns_base, **vals})
            vals[name] = orig
            grads[name] = (Cp - Cm) / (2 * h)

        for name in var_names:
            vals[name] -= lr * grads[name]

        vals_print = ", ".join(f"{name}={vals[name]:.8f}" for name in var_names)
        grad_print = ", ".join(f"{grads[name]:.8f}" for name in var_names)
        iter_text = f"iter {t}: {vals_print}, C={C:.8f}, grad=({grad_print})\n"
        out.write(iter_text)
        result['iterations'].append({
            'iter': t, 'text': iter_text.strip(),
            'loss': float(C), 'W': {n: vals[n] for n in var_names}
        })

    final_print = ", ".join(f"{name}={vals[name]}" for name in var_names)
    out.write(f"\nFinal: {final_print}\n")
    result['text'] = out.getvalue()
    result['summary_text'] = f"Final: {final_print}\n"
    return result


# ============================================================================
# GUI APPLICATION
# ============================================================================

# -- Color scheme --
SIDEBAR_BG = "#2b2d42"
SIDEBAR_FG = "#edf2f4"
SIDEBAR_ACTIVE = "#ef233c"
SIDEBAR_HOVER = "#3a3d5c"
MAIN_BG = "#f8f9fa"
ACCENT = "#2b2d42"
FONT_FAMILY = "Segoe UI"
MONO_FONT = ("Consolas", 11)
GRID_CELL_FONT = ("Consolas", 10)


# ============================================================================
# WIDGET: RoundedButton - Canvas-based button with rounded corners
# ============================================================================

class ToggleSwitch(tk.Canvas):
    """A pill-shaped toggle switch drawn on a Canvas."""

    def __init__(self, parent, variable=None, text="", font=None,
                 width=40, height=22, on_color=None, off_color="#ccc",
                 knob_color="#fff", text_color="#222", **kwargs):
        bg = parent.cget("bg") if hasattr(parent, "cget") else MAIN_BG
        super().__init__(parent, highlightthickness=0, bd=0, bg=bg, **kwargs)
        self._var = variable or tk.BooleanVar(value=False)
        self._sw = width
        self._sh = height
        self._on_color = on_color or ACCENT
        self._off_color = off_color
        self._knob_color = knob_color
        self._font = font or (FONT_FAMILY, 11)
        self._text = text

        # Measure text for total widget width
        if text:
            _tmp = tk.Label(self, text=text, font=self._font)
            _tmp.update_idletasks()
            self._tw = _tmp.winfo_reqwidth()
            self._th = _tmp.winfo_reqheight()
            _tmp.destroy()
            self._gap = 6
        else:
            self._tw = 0
            self._th = 0
            self._gap = 0

        total_w = self._sw + self._gap + self._tw
        total_h = max(self._sh, self._th)
        self.configure(width=total_w, height=total_h)

        self._draw()
        self.bind("<ButtonRelease-1>", self._on_click)
        self._var.trace_add("write", lambda *a: self._draw())

    def _draw(self):
        self.delete("all")
        on = self._var.get()
        sw, sh = self._sw, self._sh
        total_h = int(self.cget("height"))
        y_off = (total_h - sh) // 2
        r = sh // 2
        pad = 3
        knob_r = r - pad

        # Track color
        color = self._on_color if on else self._off_color

        # Draw pill track
        self.create_arc(0, y_off, sh, y_off + sh, start=90, extent=180, fill=color, outline=color)
        self.create_arc(sw - sh, y_off, sw, y_off + sh, start=270, extent=180, fill=color, outline=color)
        self.create_rectangle(r, y_off, sw - r, y_off + sh, fill=color, outline=color)

        # Draw knob
        if on:
            cx = sw - r
        else:
            cx = r
        cy = y_off + r
        self.create_oval(cx - knob_r, cy - knob_r, cx + knob_r, cy + knob_r,
                         fill=self._knob_color, outline=self._knob_color)

        # Draw label text
        if self._text:
            tx = sw + self._gap
            ty = total_h // 2
            self.create_text(tx, ty, text=self._text, font=self._font,
                             fill="#222", anchor="w")

    def _on_click(self, event):
        self._var.set(not self._var.get())


class RoundedButton(tk.Canvas):
    """A button with rounded corners drawn on a Canvas."""

    def __init__(self, parent, text="", command=None, width=None, radius=6,
                 font=None, bg_color="#e0e0e0", fg_color="#222", hover_color="#c8c8c8",
                 press_color="#b0b0b0", padx=6, pady=2, **kwargs):
        super().__init__(parent, highlightthickness=0, bd=0,
                         bg=parent.cget("bg") if hasattr(parent, "cget") else MAIN_BG,
                         **kwargs)
        self._text = text
        self._command = command
        self._radius = radius
        self._bg_color = bg_color
        self._fg_color = fg_color
        self._hover_color = hover_color
        self._press_color = press_color
        self._padx = padx
        self._pady = pady
        self._font = font or (FONT_FAMILY, 10)
        self._disabled = False

        # Measure text to determine canvas size
        _tmp = tk.Label(self, text=text, font=self._font)
        _tmp.update_idletasks()
        tw = _tmp.winfo_reqwidth()
        th = _tmp.winfo_reqheight()
        _tmp.destroy()

        if width is not None:
            # width in approximate character widths
            cw = tk.Label(self, text="0", font=self._font)
            cw.update_idletasks()
            char_w = cw.winfo_reqwidth()
            cw.destroy()
            tw = max(tw, int(width * char_w * 0.75))

        self._btn_w = tw + padx * 2
        self._btn_h = th + pady * 2
        self.configure(width=self._btn_w, height=self._btn_h)

        self._draw(self._bg_color)

        self.bind("<Enter>", lambda e: self._draw(self._hover_color) if not self._disabled else None)
        self.bind("<Leave>", lambda e: self._draw(self._bg_color) if not self._disabled else None)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw(self, fill):
        self.delete("all")
        r = self._radius
        w, h = self._btn_w, self._btn_h
        # Rounded rectangle via arcs + rectangles
        self.create_arc(0, 0, r * 2, r * 2, start=90, extent=90, fill=fill, outline=fill)
        self.create_arc(w - r * 2, 0, w, r * 2, start=0, extent=90, fill=fill, outline=fill)
        self.create_arc(0, h - r * 2, r * 2, h, start=180, extent=90, fill=fill, outline=fill)
        self.create_arc(w - r * 2, h - r * 2, w, h, start=270, extent=90, fill=fill, outline=fill)
        self.create_rectangle(r, 0, w - r, h, fill=fill, outline=fill)
        self.create_rectangle(0, r, w, h - r, fill=fill, outline=fill)
        self.create_text(w // 2, h // 2, text=self._text, font=self._font, fill=self._fg_color)

    def _on_press(self, event):
        if not self._disabled:
            self._draw(self._press_color)

    def _on_release(self, event):
        if not self._disabled:
            self._draw(self._hover_color)
            if self._command and 0 <= event.x <= self._btn_w and 0 <= event.y <= self._btn_h:
                self._command()

    def configure_state(self, state):
        self._disabled = (state == "disabled")


# ============================================================================
# WIDGET: MatrixGrid - Calculator-style matrix input
# ============================================================================

class MatrixGrid(tk.Frame):
    """A graphing-calculator-style grid widget for matrix/vector input."""

    def __init__(self, parent, label, rows=2, cols=2, vector_mode=False,
                 row_label="samples", col_label="features", on_resize=None,
                 hide_rows=False, hide_cols=False):
        super().__init__(parent, bg=MAIN_BG)
        self.label = label
        self.n_rows = rows
        self.n_cols = cols
        self.vector_mode = vector_mode
        self.row_label = row_label
        self.col_label = col_label
        self.cells = []  # 2D list of Entry widgets
        self.on_resize = on_resize  # callback(new_rows, new_cols)

        # Header row: label + dimension controls
        header = tk.Frame(self, bg=MAIN_BG)
        header.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(header, text=label, font=(FONT_FAMILY, 10, "bold")).pack(side=tk.LEFT)

        self.rows_var = tk.StringVar(value=str(rows))
        if not vector_mode:
            if not hide_rows:
                ttk.Label(header, text="Rows:").pack(side=tk.LEFT, padx=(8, 0))
                rows_spin = ttk.Spinbox(header, from_=1, to=50, textvariable=self.rows_var,
                                        width=3, command=self._on_spin_change)
                rows_spin.pack(side=tk.LEFT, padx=(2, 4))

            if not hide_cols:
                ttk.Label(header, text="Cols:").pack(side=tk.LEFT, padx=(8, 0) if hide_rows else (0, 0))
            self.cols_var = tk.StringVar(value=str(cols))
            if not hide_cols:
                cols_spin = ttk.Spinbox(header, from_=1, to=50, textvariable=self.cols_var,
                                        width=3, command=self._on_spin_change)
                cols_spin.pack(side=tk.LEFT, padx=(2, 4))
        else:
            ttk.Label(header, text="Size:").pack(side=tk.LEFT, padx=(8, 0))
            rows_spin = ttk.Spinbox(header, from_=1, to=50, textvariable=self.rows_var,
                                    width=3, command=self._on_spin_change)
            rows_spin.pack(side=tk.LEFT, padx=(2, 4))
            self.cols_var = tk.StringVar(value="1")

        RoundedButton(header, text="Paste", width=5, command=self._paste_from_clipboard,
                      font=(FONT_FAMILY, 9), pady=1).pack(side=tk.LEFT, padx=(6, 0))
        RoundedButton(header, text="Copy", width=5, command=self._copy_to_clipboard,
                      font=(FONT_FAMILY, 9), pady=1).pack(side=tk.LEFT, padx=(2, 0))
        RoundedButton(header, text="Clear", width=5, command=self._clear_all,
                      font=(FONT_FAMILY, 9), pady=1).pack(side=tk.LEFT, padx=(2, 0))

        # Grid area
        self.grid_outer = tk.Frame(self, bg="#e8e8e8", bd=1, relief=tk.SOLID)
        self.grid_outer.pack(anchor="w", pady=2)

        self.grid_inner = tk.Frame(self.grid_outer, bg="#f0f0f0")
        self.grid_inner.pack()

        # Shape label
        self.shape_label = ttk.Label(self, text="", foreground="#888", font=(FONT_FAMILY, 8))
        self.shape_label.pack(anchor="w")

        self._build_grid()

    @staticmethod
    def _validate_cell(new_value):
        """Allow only digits, dots, minus signs, and empty string."""
        if new_value == "":
            return True
        for ch in new_value:
            if ch not in "0123456789.-":
                return False
        return True

    def _on_cell_focus_out(self, event):
        ent = event.widget
        if not ent.get().strip():
            ent.insert(0, "0")

    def _build_grid(self):
        for widget in self.grid_inner.winfo_children():
            widget.destroy()
        self.cells = []

        vcmd = (self.register(self._validate_cell), '%P')

        for r in range(self.n_rows):
            row_cells = []
            # Left bracket
            tk.Label(self.grid_inner, text="[", font=("Consolas", 14, "bold"),
                     bg="#f0f0f0", fg="#555").grid(row=r, column=0, padx=(4, 0))
            for c in range(self.n_cols):
                ent = tk.Entry(self.grid_inner, width=8, font=GRID_CELL_FONT,
                               justify=tk.CENTER, relief=tk.SOLID, bd=1,
                               validate="key", validatecommand=vcmd)
                ent.grid(row=r, column=c + 1, padx=1, pady=1)
                ent.insert(0, "0")
                ent.bind("<Control-v>", self._on_paste)
                ent.bind("<<Paste>>", self._on_paste)
                ent.bind("<FocusOut>", self._on_cell_focus_out)
                row_cells.append(ent)
            # Right bracket
            tk.Label(self.grid_inner, text="]", font=("Consolas", 14, "bold"),
                     bg="#f0f0f0", fg="#555").grid(row=r, column=self.n_cols + 1, padx=(0, 4))
            self.cells.append(row_cells)

        self._update_shape_label()

    def _on_spin_change(self):
        try:
            new_rows = int(self.rows_var.get())
            new_cols = int(self.cols_var.get()) if not self.vector_mode else 1
        except ValueError:
            return
        new_rows = max(1, min(50, new_rows))
        new_cols = max(1, min(50, new_cols))
        self._resize(new_rows, new_cols)

    def _resize(self, new_rows, new_cols):
        # Save existing values
        old_vals = []
        for r in range(min(self.n_rows, new_rows)):
            row_vals = []
            for c in range(min(self.n_cols, new_cols)):
                row_vals.append(self.cells[r][c].get())
            old_vals.append(row_vals)

        self.n_rows = new_rows
        self.n_cols = new_cols
        self.rows_var.set(str(new_rows))
        if not self.vector_mode:
            self.cols_var.set(str(new_cols))
        self._build_grid()

        # Restore saved values
        for r in range(len(old_vals)):
            for c in range(len(old_vals[r])):
                self.cells[r][c].delete(0, tk.END)
                self.cells[r][c].insert(0, old_vals[r][c])

        if self.on_resize:
            self.on_resize(new_rows, new_cols)

    def _add_row(self):
        self._resize(self.n_rows + 1, self.n_cols)

    def _remove_row(self):
        if self.n_rows > 1:
            self._resize(self.n_rows - 1, self.n_cols)

    def _add_col(self):
        self._resize(self.n_rows, self.n_cols + 1)

    def _remove_col(self):
        if self.n_cols > 1:
            self._resize(self.n_rows, self.n_cols - 1)

    def _clear_all(self):
        for row in self.cells:
            for ent in row:
                ent.delete(0, tk.END)
                ent.insert(0, "0")

    def _parse_paste_data(self, text):
        """Parse clipboard text into a 2D list of strings."""
        text = text.strip()
        if not text:
            return [["0"]]

        # Try comma-separated rows format: "1 2, 3 4, 5 6"
        if ',' in text and '\t' not in text:
            rows = [r.strip() for r in text.split(',') if r.strip()]
            data = []
            for r in rows:
                cols = r.split()
                data.append(cols)
            return data

        # Tab-separated (Excel paste)
        if '\t' in text:
            rows = text.split('\n')
            data = []
            for r in rows:
                r = r.strip()
                if r:
                    data.append(r.split('\t'))
            return data

        # Newline-separated rows, space-separated columns
        if '\n' in text:
            rows = text.split('\n')
            data = []
            for r in rows:
                r = r.strip()
                if r:
                    data.append(r.split())
            return data

        # Single row, space-separated
        return [text.split()]

    def _on_paste(self, event=None):
        try:
            text = self.winfo_toplevel().clipboard_get()
        except tk.TclError:
            return
        data = self._parse_paste_data(text)
        if not data:
            return

        new_rows = len(data)
        new_cols = max(len(r) for r in data)
        if self.vector_mode:
            # For vectors, paste as single column
            if new_cols == 1:
                self._resize(new_rows, 1)
            else:
                # Flatten to column
                flat = []
                for r in data:
                    flat.extend(r)
                data = [[v] for v in flat]
                new_rows = len(flat)
                new_cols = 1
                self._resize(new_rows, 1)
        else:
            self._resize(new_rows, new_cols)

        for r in range(new_rows):
            for c in range(new_cols):
                if r < len(data) and c < len(data[r]):
                    self.cells[r][c].delete(0, tk.END)
                    self.cells[r][c].insert(0, data[r][c])
        return "break"

    def _paste_from_clipboard(self):
        self._on_paste()

    def _copy_to_clipboard(self):
        """Copy grid contents to clipboard as tab-separated rows."""
        lines = []
        for row in self.cells:
            lines.append("\t".join(ent.get().strip() or "0" for ent in row))
        text = "\n".join(lines)
        self.winfo_toplevel().clipboard_clear()
        self.winfo_toplevel().clipboard_append(text)

    def _update_shape_label(self):
        if self.vector_mode:
            self.shape_label.config(text=f"{self.n_rows} {self.row_label}")
        else:
            self.shape_label.config(
                text=f"{self.n_rows} {self.row_label} \u00d7 {self.n_cols} {self.col_label}")

    def get_matrix_string(self):
        """Return comma-separated row format: '1 2, 3 4, 5 6'."""
        rows = []
        for row in self.cells:
            vals = []
            for ent in row:
                v = ent.get().strip()
                if not v:
                    v = "0"
                vals.append(v)
            rows.append(" ".join(vals))
        return ", ".join(rows)

    def get_vector_string(self):
        """Return space-separated values for 1D vectors."""
        vals = []
        for row in self.cells:
            for ent in row:
                v = ent.get().strip()
                if not v:
                    v = "0"
                vals.append(v)
        return " ".join(vals)

    def set_from_string(self, s):
        """Populate grid from comma-separated row string."""
        s = s.strip()
        if not s:
            return
        data = self._parse_paste_data(s)
        new_rows = len(data)
        new_cols = max(len(r) for r in data)
        if self.vector_mode:
            new_cols = 1
        self._resize(new_rows, new_cols)
        for r in range(new_rows):
            for c in range(new_cols):
                if r < len(data) and c < len(data[r]):
                    self.cells[r][c].delete(0, tk.END)
                    self.cells[r][c].insert(0, data[r][c])

    def set_from_matrix(self, arr):
        """Resize grid to match arr shape and fill cells from a numpy array."""
        arr = np.atleast_2d(arr)
        self._resize(arr.shape[0], arr.shape[1])
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                self.cells[r][c].delete(0, tk.END)
                self.cells[r][c].insert(0, str(round(float(arr[r, c]), 8)))

    def get_shape(self):
        return (self.n_rows, self.n_cols)


# ============================================================================
# WIDGET: ExpressionEditor - Text input with embedded variable chips
# ============================================================================

_SUBSCRIPTS = str.maketrans('12345', '\u2081\u2082\u2083\u2084\u2085')


class ExpressionEditor(tk.Frame):
    """Single-line expression editor where variables are embedded chip widgets,
    insertable only via button press."""

    def __init__(self, parent, font=None, **kwargs):
        super().__init__(parent, bg=MAIN_BG)
        self._font = font or MONO_FONT
        self._text = tk.Text(self, height=1, wrap=tk.NONE, font=self._font,
                             relief=tk.SOLID, bd=1, undo=True)
        self._text.pack(fill=tk.X)
        self._text.bind("<Return>", lambda e: "break")
        self._text.bind("<KeyRelease>", lambda e: self._fire_change())
        self._text.bind("<<Paste>>", lambda e: self.after(10, self._fire_change))
        self._chip_map = {}  # str(widget) -> var_name
        self._on_change_cb = None

    # -- public API used by MathKeyboard (Entry-compatible shims) --

    def insert(self, index, text):
        self._text.insert(index, text)
        self._fire_change()

    def index(self, idx):
        """Return integer column (single-line assumption)."""
        s = self._text.index(idx)
        return int(s.split('.')[1])

    def icursor(self, pos):
        self._text.mark_set(tk.INSERT, f"1.{pos}")

    def focus_set(self):
        self._text.focus_set()

    # -- chip insertion --

    def insert_variable(self, var_name):
        """Insert a variable chip at the cursor position."""
        display = 'x' + var_name[1:].translate(_SUBSCRIPTS)
        chip = tk.Label(self._text, text=display,
                        bg="#dce6f7", fg="#1a56db",
                        font=(FONT_FAMILY, 9),
                        padx=2, pady=0, bd=0)
        self._text.window_create(tk.INSERT, window=chip, padx=1)
        self._chip_map[str(chip)] = var_name
        self._text.focus_set()
        self._fire_change()

    # -- expression extraction --

    def get_expression(self):
        """Return the expression string with chip positions replaced by var names."""
        parts = []
        try:
            for key, val, idx in self._text.dump("1.0", "end-1c",
                                                  text=True, window=True):
                if key == 'text':
                    parts.append(val)
                elif key == 'window':
                    name = self._chip_map.get(val)
                    if name:
                        parts.append(name)
        except tk.TclError:
            pass
        return ''.join(parts).replace('^', '**')

    def get(self, *args):
        """Compat shim: return full expression (^ converted to **)."""
        return self.get_expression()

    def get_variables(self):
        """Return sorted list of unique variable names currently present as chips."""
        found = set()
        try:
            for key, val, idx in self._text.dump("1.0", "end-1c", window=True):
                if key == 'window':
                    name = self._chip_map.get(val)
                    if name:
                        found.add(name)
        except tk.TclError:
            pass
        return sorted(found)

    # -- change callback --

    def set_on_change(self, callback):
        self._on_change_cb = callback

    def _fire_change(self):
        if self._on_change_cb:
            self._on_change_cb()


# ============================================================================
# WIDGET: MathKeyboard - Desmos-style math button pad
# ============================================================================

class MathKeyboard(tk.Frame):
    """Clickable math function button pad for the Cost Minimizer."""

    BUTTONS = [
        # (label, insert_text, cursor_back)
        ("sin", "sin()", 1), ("cos", "cos()", 1), ("tan", "tan()", 1),
        ("exp", "exp()", 1), ("ln", "ln()", 1), ("sqrt", "sqrt()", 1),
        ("\u03c0", "pi", 0), ("e", "e", 0),
        ("^", "^", 0), ("(", "(", 0), (")", ")", 0),
        ("*", "*", 0), ("/", "/", 0), ("+", "+", 0),
        ("-", "-", 0), ("^2", "^2", 0),
    ]

    def __init__(self, parent, target_widget):
        super().__init__(parent, bg=MAIN_BG)
        self.target = target_widget
        for i, (label, text, back) in enumerate(self.BUTTONS):
            btn = RoundedButton(self, text=label, width=5,
                                command=lambda t=text, b=back: self._insert(t, b),
                                font=(FONT_FAMILY, 9), pady=1)
            btn.grid(row=i // 8, column=i % 8, padx=1, pady=1)

    def _insert(self, text, cursor_back):
        self.target.insert(tk.INSERT, text)
        if cursor_back > 0:
            pos = self.target.index(tk.INSERT)
            self.target.icursor(pos - cursor_back)
        self.target.focus_set()


# ============================================================================
# WIDGET: CollapsibleSection - Expandable iteration block
# ============================================================================

class CollapsibleSection(tk.Frame):
    """A collapsible section with a clickable header and hideable body."""

    def __init__(self, parent, title, body_text, expanded=False):
        super().__init__(parent, bg="#1e1e2e")
        self._expanded = expanded

        # Header bar
        self._header = tk.Frame(self, bg="#2a2a3e", cursor="hand2")
        self._header.pack(fill=tk.X, pady=(1, 0))

        self._arrow = tk.Label(self._header, text="\u25bc" if expanded else "\u25b6",
                               bg="#2a2a3e", fg="#89b4fa", font=("Consolas", 10))
        self._arrow.pack(side=tk.LEFT, padx=(6, 4))

        self._title_lbl = tk.Label(self._header, text=title, bg="#2a2a3e", fg="#cdd6f4",
                                   font=("Consolas", 10), anchor="w")
        self._title_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Body
        self._body = tk.Text(self, font=MONO_FONT, wrap=tk.WORD, height=min(12, body_text.count('\n') + 2),
                             bg="#1e1e2e", fg="#cdd6f4", insertbackground="#cdd6f4",
                             selectbackground="#45475a", relief=tk.FLAT, padx=8, pady=4)
        self._body.insert("1.0", body_text)
        self._body.configure(state=tk.DISABLED)

        if expanded:
            self._body.pack(fill=tk.X, padx=(16, 0))

        # Bind click
        for w in (self._header, self._arrow, self._title_lbl):
            w.bind("<Button-1>", self._toggle)

    def _toggle(self, event=None):
        self._expanded = not self._expanded
        if self._expanded:
            self._arrow.configure(text="\u25bc")
            self._body.pack(fill=tk.X, padx=(16, 0))
        else:
            self._arrow.configure(text="\u25b6")
            self._body.pack_forget()


# ============================================================================
# WIDGET: TreeVisualizer - Canvas-based decision tree drawing
# ============================================================================

class TreeVisualizer(tk.Frame):
    """Draws a decision tree on a scrollable Canvas."""

    NODE_W = 140
    NODE_H = 40
    H_GAP = 20
    V_GAP = 60
    INTERNAL_COLOR = "#89b4fa"
    LEAF_COLOR = "#a6e3a1"
    TEXT_COLOR = "#1e1e2e"
    LINE_COLOR = "#6c7086"

    def __init__(self, parent, tree_root):
        super().__init__(parent, bg="#1e1e2e")
        self._tree = tree_root

        self._canvas = tk.Canvas(self, bg="#1e1e2e", highlightthickness=0)
        h_scroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self._canvas.xview)
        v_scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        self._canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._positions = {}
        self._draw_tree()

    def _count_leaves(self, node):
        if node.leaf():
            return 1
        return self._count_leaves(node.l) + self._count_leaves(node.r)

    def _compute_positions(self, node, depth, left, right):
        if node.leaf():
            x = (left + right) / 2
            y = depth * (self.NODE_H + self.V_GAP) + 30
            self._positions[id(node)] = (x, y)
            return

        n_left = self._count_leaves(node.l)
        n_total = self._count_leaves(node.l) + self._count_leaves(node.r)
        split = left + (right - left) * n_left / n_total

        self._compute_positions(node.l, depth + 1, left, split)
        self._compute_positions(node.r, depth + 1, split, right)

        lx, ly = self._positions[id(node.l)]
        rx, ry = self._positions[id(node.r)]
        x = (lx + rx) / 2
        y = depth * (self.NODE_H + self.V_GAP) + 30
        self._positions[id(node)] = (x, y)

    def _draw_node(self, node):
        x, y = self._positions[id(node)]
        hw, hh = self.NODE_W // 2, self.NODE_H // 2

        if node.leaf():
            color = self.LEAF_COLOR
            pred_val = node.pred
            if isinstance(pred_val, (float, np.floating)):
                text = f"pred={pred_val:.4f}"
            else:
                text = f"pred={pred_val}"
        else:
            color = self.INTERNAL_COLOR
            text = f"X[{node.f}] <= {node.t:.4f}"

        # Rounded rectangle
        r = 8
        self._canvas.create_arc(x - hw, y - hh, x - hw + 2 * r, y - hh + 2 * r,
                                start=90, extent=90, fill=color, outline=color)
        self._canvas.create_arc(x + hw - 2 * r, y - hh, x + hw, y - hh + 2 * r,
                                start=0, extent=90, fill=color, outline=color)
        self._canvas.create_arc(x - hw, y + hh - 2 * r, x - hw + 2 * r, y + hh,
                                start=180, extent=90, fill=color, outline=color)
        self._canvas.create_arc(x + hw - 2 * r, y + hh - 2 * r, x + hw, y + hh,
                                start=270, extent=90, fill=color, outline=color)
        self._canvas.create_rectangle(x - hw + r, y - hh, x + hw - r, y + hh,
                                      fill=color, outline=color)
        self._canvas.create_rectangle(x - hw, y - hh + r, x + hw, y + hh - r,
                                      fill=color, outline=color)
        self._canvas.create_text(x, y, text=text, font=("Consolas", 9),
                                 fill=self.TEXT_COLOR)

        if not node.leaf():
            # Draw edges to children
            for child, label in [(node.l, "Yes"), (node.r, "No")]:
                cx, cy = self._positions[id(child)]
                self._canvas.create_line(x, y + hh, cx, cy - hh,
                                         fill=self.LINE_COLOR, width=2)
                mx, my = (x + cx) / 2, (y + hh + cy - hh) / 2
                self._canvas.create_text(mx, my - 8, text=label,
                                         font=("Consolas", 8), fill="#bac2de")
            self._draw_node(node.l)
            self._draw_node(node.r)

    def _draw_tree(self):
        if self._tree is None:
            return
        n_leaves = self._count_leaves(self._tree)
        total_w = max(400, n_leaves * (self.NODE_W + self.H_GAP))
        self._compute_positions(self._tree, 0, 0, total_w)
        self._draw_node(self._tree)

        # Set scroll region
        self._canvas.configure(scrollregion=self._canvas.bbox("all") or (0, 0, 400, 300))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EE2211 Exam Toolkit")
        self.geometry("1280x950")
        self.minsize(900, 650)
        self.configure(bg=MAIN_BG)

        # DPI scaling
        try:
            dpi = self.winfo_fpixels('1i')
            scale_factor = dpi / 72.0
            self.tk.call('tk', 'scaling', scale_factor)
        except Exception:
            pass

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", background=MAIN_BG, font=(FONT_FAMILY, 11))
        style.configure("Header.TLabel", font=(FONT_FAMILY, 16, "bold"), background=MAIN_BG)
        style.configure("TCombobox", font=(FONT_FAMILY, 11))
        style.configure("TSpinbox", arrowsize=17)

        # Sidebar
        self.sidebar = tk.Frame(self, bg=SIDEBAR_BG, width=240)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="EE2211", bg=SIDEBAR_BG, fg=SIDEBAR_FG,
                 font=(FONT_FAMILY, 16, "bold"), pady=12).pack(fill=tk.X)
        tk.Frame(self.sidebar, bg=SIDEBAR_ACTIVE, height=2).pack(fill=tk.X, padx=10, pady=(0, 8))

        # Main content
        self.main_frame = tk.Frame(self, bg=MAIN_BG)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.modules = [
            ("Classification", ClassificationFrame),
            ("Clustering", ClusteringFrame),
            ("Gradient Desc", GradientDescentFrame),
            ("Neural Net", NeuralNetFrame),
            ("Regression", RegressionFrame),
            ("Decision Tree", DecisionTreeFrame),
            ("Cost Minimizer", CostMinimizerFrame),
        ]

        self.sidebar_buttons = []
        self.frames = {}
        self.active_name = None

        for name, frame_cls in self.modules:
            btn = tk.Label(self.sidebar, text=f"  {name}", bg=SIDEBAR_BG, fg=SIDEBAR_FG,
                           font=(FONT_FAMILY, 11), anchor="w", padx=12, pady=8, cursor="hand2")
            btn.pack(fill=tk.X)
            btn.bind("<Button-1>", lambda e, n=name: self.show_frame(n))
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=SIDEBAR_HOVER) if b != self._active_btn() else None)
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg=SIDEBAR_BG) if b != self._active_btn() else None)
            self.sidebar_buttons.append((name, btn))

            frame = frame_cls(self.main_frame)
            self.frames[name] = frame

        # Exit button at bottom
        tk.Frame(self.sidebar, bg=SIDEBAR_BG).pack(fill=tk.BOTH, expand=True)
        exit_btn = tk.Label(self.sidebar, text="  Exit", bg=SIDEBAR_BG, fg="#aaa",
                            font=(FONT_FAMILY, 11), anchor="w", padx=12, pady=8, cursor="hand2")
        exit_btn.pack(fill=tk.X, side=tk.BOTTOM, pady=(0, 10))
        exit_btn.bind("<Button-1>", lambda e: self.destroy())

        self.show_frame("Classification")
        self.bind("<Control-Return>", self._run_active)

    def _active_btn(self):
        for name, btn in self.sidebar_buttons:
            if name == self.active_name:
                return btn
        return None

    def _run_active(self, event=None):
        if self.active_name and self.active_name in self.frames:
            self.frames[self.active_name].run()

    def show_frame(self, name):
        for n, btn in self.sidebar_buttons:
            if n == name:
                btn.configure(bg=SIDEBAR_ACTIVE, fg="white")
            else:
                btn.configure(bg=SIDEBAR_BG, fg=SIDEBAR_FG)
        for n, f in self.frames.items():
            f.pack_forget()
        self.frames[name].pack(fill=tk.BOTH, expand=True, padx=16, pady=10)
        self.active_name = name


# ============================================================================
# BASE MODULE FRAME
# ============================================================================

class ModuleFrame(tk.Frame):
    """Base class for all module frames."""

    def __init__(self, parent, title):
        super().__init__(parent, bg=MAIN_BG)
        self._last_result = None
        self.title_label = ttk.Label(self, text=title, style="Header.TLabel")
        self.title_label.pack(anchor="w", pady=(0, 8))

        # Paned window: top = inputs, bottom = output
        self.pane = ttk.PanedWindow(self, orient=tk.VERTICAL)
        self.pane.pack(fill=tk.BOTH, expand=True)

        # Input area (scrollable, no visible scrollbar)
        self.input_outer = tk.Frame(self.pane, bg=MAIN_BG)
        self.input_canvas = tk.Canvas(self.input_outer, bg=MAIN_BG, highlightthickness=0)
        self.input_frame = tk.Frame(self.input_canvas, bg=MAIN_BG)

        self.input_frame.bind("<Configure>", lambda e: self.input_canvas.configure(
            scrollregion=self.input_canvas.bbox("all")))
        self.canvas_window = self.input_canvas.create_window((0, 0), window=self.input_frame, anchor="nw")
        self.input_canvas.bind("<Configure>", lambda e: self.input_canvas.itemconfig(
            self.canvas_window, width=e.width))

        self.input_canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mousewheel to input canvas
        self.input_frame.bind("<Enter>", lambda e: self._bind_mousewheel(self.input_canvas))
        self.input_frame.bind("<Leave>", lambda e: self._unbind_mousewheel(self.input_canvas))

        self.pane.add(self.input_outer, weight=1)

        # Button bar
        self._btn_frame = tk.Frame(self, bg=MAIN_BG)
        self._btn_frame.pack(fill=tk.X, pady=4)
        RoundedButton(self._btn_frame, text="Run", command=self.run,
                      font=(FONT_FAMILY, 12, "bold"), bg_color=ACCENT, fg_color="#fff",
                      hover_color="#3a3d5c", press_color="#1a1d32",
                      padx=14, pady=4).pack(side=tk.LEFT, padx=(0, 8))
        RoundedButton(self._btn_frame, text="Clear Output", command=self.clear_output
                      ).pack(side=tk.LEFT)

        # Output area with tabbed notebook
        self.output_frame = tk.Frame(self, bg=MAIN_BG)
        self.output_frame.pack(fill=tk.BOTH, expand=True)

        # Style the notebook tabs to match the dark theme
        style = ttk.Style()
        style.configure("Dark.TNotebook", background=MAIN_BG)
        style.configure("Dark.TNotebook.Tab", background="#2a2a3e", foreground="#cdd6f4",
                        font=("Consolas", 10), padding=[10, 4])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", "#1e1e2e"), ("!selected", "#2a2a3e")],
                  foreground=[("selected", "#89b4fa"), ("!selected", "#cdd6f4")])

        self.output_notebook = ttk.Notebook(self.output_frame, style="Dark.TNotebook")
        self.output_notebook.pack(fill=tk.BOTH, expand=True)

        # Summary tab (always present)
        self.summary_tab = tk.Text(self.output_notebook, font=MONO_FONT, wrap=tk.WORD, height=12,
                                   bg="#1e1e2e", fg="#cdd6f4", insertbackground="#cdd6f4",
                                   selectbackground="#45475a", relief=tk.FLAT, padx=8, pady=6)
        self.output_notebook.add(self.summary_tab, text="Summary")

        # Keep reference to old output for backward compat
        self.output = self.summary_tab

        # Bind mousewheel to summary
        self.summary_tab.bind("<Enter>", lambda e: self._bind_mousewheel(self.summary_tab))
        self.summary_tab.bind("<Leave>", lambda e: self._unbind_mousewheel(self.summary_tab))

        # Iterations tab (created on demand)
        self._iterations_tab = None
        self._iterations_canvas = None
        self._iterations_inner = None

        # Weights tab (created on demand)
        self._weights_tab = None

        # Tree tab (created on demand)
        self._tree_tab = None

    def _bind_mousewheel(self, widget):
        widget.bind_all("<MouseWheel>", lambda e: self._on_mousewheel(e, widget))

    def _unbind_mousewheel(self, widget):
        widget.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event, widget):
        if isinstance(widget, tk.Canvas):
            widget.yview_scroll(int(-1 * (event.delta / 120)), "units")
        else:
            widget.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _remove_tab(self, tab_widget):
        """Remove a tab from the notebook if it exists."""
        if tab_widget is not None:
            try:
                self.output_notebook.forget(tab_widget)
            except tk.TclError:
                pass

    def _build_iterations_tab(self, iterations):
        """Build or rebuild the Iterations tab with collapsible sections."""
        self._remove_tab(self._iterations_tab)

        outer = tk.Frame(self.output_notebook, bg="#1e1e2e")

        # Scrollable area
        canvas = tk.Canvas(outer, bg="#1e1e2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg="#1e1e2e")

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_win, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mousewheel binding
        inner.bind("<Enter>", lambda e: self._bind_mousewheel(canvas))
        inner.bind("<Leave>", lambda e: self._unbind_mousewheel(canvas))

        # Determine which iterations to show
        n = len(iterations)
        show_all_ref = [False]
        max_collapsed = 50

        def _populate(show_all=False):
            for w in inner.winfo_children():
                w.destroy()

            if n <= max_collapsed or show_all:
                items = iterations
            else:
                items = iterations[:10] + [None] + iterations[-10:]

            for item in items:
                if item is None:
                    # "Show all" button
                    btn_frame = tk.Frame(inner, bg="#1e1e2e")
                    btn_frame.pack(fill=tk.X, pady=4, padx=8)
                    RoundedButton(btn_frame, text=f"Show all {n} iterations",
                                  command=lambda: _populate(True),
                                  font=(FONT_FAMILY, 10), padx=12, pady=3,
                                  bg_color="#45475a", fg_color="#cdd6f4",
                                  hover_color="#585b70", press_color="#6c7086"
                                  ).pack(anchor="w")
                    continue

                loss_str = f" | loss={item['loss']:.6f}" if 'loss' in item else ""
                title = f"Iteration {item['iter']}{loss_str}"
                section = CollapsibleSection(inner, title, item['text'], expanded=False)
                section.pack(fill=tk.X, padx=4, pady=1)

        _populate()

        self._iterations_tab = outer
        self._iterations_canvas = canvas
        self._iterations_inner = inner
        self.output_notebook.add(outer, text="Iterations")

    def _build_weights_tab(self, result):
        """Build or rebuild the Weights tab showing weight matrices."""
        self._remove_tab(self._weights_tab)

        outer = tk.Frame(self.output_notebook, bg="#1e1e2e")
        weights = result.get('weights')
        bias = result.get('bias')

        if weights is None:
            self._weights_tab = None
            return

        # Scrollable text for weight display
        text_w = tk.Text(outer, font=MONO_FONT, wrap=tk.WORD,
                         bg="#1e1e2e", fg="#cdd6f4", insertbackground="#cdd6f4",
                         selectbackground="#45475a", relief=tk.FLAT, padx=8, pady=6)
        text_w.pack(fill=tk.BOTH, expand=True)

        if isinstance(weights, list):
            # Neural network: list of weight matrices
            for i, w in enumerate(weights, 1):
                text_w.insert(tk.END, f"W^{i} (shape {w.shape}):\n{np.round(w, 8)}\n\n")
        else:
            w_arr = np.atleast_2d(weights)
            text_w.insert(tk.END, f"W (shape {w_arr.shape}):\n{np.round(w_arr, 8)}\n\n")

        if bias is not None:
            b_arr = np.atleast_1d(bias)
            text_w.insert(tk.END, f"b (intercept): {np.round(b_arr, 8)}\n")

        text_w.configure(state=tk.DISABLED)

        self._weights_tab = outer
        self.output_notebook.add(outer, text="Weights")

    def _build_tree_tab(self, tree_node):
        """Build or rebuild the Tree visualization tab."""
        self._remove_tab(self._tree_tab)
        if tree_node is None:
            self._tree_tab = None
            return

        viz = TreeVisualizer(self.output_notebook, tree_node)
        self._tree_tab = viz
        self.output_notebook.add(viz, text="Tree")

    def show_result(self, result_dict):
        """Populate tabs from a structured result dict."""
        self._last_result = result_dict

        # Summary tab
        summary_text = result_dict.get('summary_text', result_dict.get('text', ''))
        self.summary_tab.configure(state=tk.NORMAL)
        self.summary_tab.delete("1.0", tk.END)
        self.summary_tab.insert(tk.END, summary_text)
        self.summary_tab.configure(state=tk.DISABLED)

        # Iterations tab
        iterations = result_dict.get('iterations', [])
        self._remove_tab(self._iterations_tab)
        self._iterations_tab = None
        if iterations:
            self._build_iterations_tab(iterations)

        # Weights tab
        self._remove_tab(self._weights_tab)
        self._weights_tab = None
        if result_dict.get('weights') is not None:
            self._build_weights_tab(result_dict)

        # Tree tab
        self._remove_tab(self._tree_tab)
        self._tree_tab = None
        if result_dict.get('tree') is not None:
            self._build_tree_tab(result_dict['tree'])

        # Select Summary tab
        self.output_notebook.select(self.summary_tab)

    def show_output(self, text):
        """Backward-compatible: accept string or dict."""
        if isinstance(text, dict):
            self.show_result(text)
            return
        self._last_result = None
        # Remove dynamic tabs
        self._remove_tab(self._iterations_tab)
        self._iterations_tab = None
        self._remove_tab(self._weights_tab)
        self._weights_tab = None
        self._remove_tab(self._tree_tab)
        self._tree_tab = None

        self.summary_tab.configure(state=tk.NORMAL)
        self.summary_tab.delete("1.0", tk.END)
        self.summary_tab.insert(tk.END, text)
        self.summary_tab.configure(state=tk.DISABLED)
        self.summary_tab.see(tk.END)

    def clear_output(self):
        self._last_result = None
        self.summary_tab.configure(state=tk.NORMAL)
        self.summary_tab.delete("1.0", tk.END)
        self.summary_tab.configure(state=tk.DISABLED)
        self._remove_tab(self._iterations_tab)
        self._iterations_tab = None
        self._remove_tab(self._weights_tab)
        self._weights_tab = None
        self._remove_tab(self._tree_tab)
        self._tree_tab = None

    def copy_output(self):
        """Copy from the currently active tab."""
        self.clipboard_clear()
        try:
            current = self.output_notebook.nametowidget(self.output_notebook.select())
        except (tk.TclError, KeyError):
            current = self.summary_tab

        if isinstance(current, tk.Text):
            self.clipboard_append(current.get("1.0", tk.END))
        elif hasattr(current, 'winfo_children'):
            # Try to find a Text widget inside
            for child in current.winfo_children():
                if isinstance(child, tk.Text):
                    self.clipboard_append(child.get("1.0", tk.END))
                    return
            # Fallback: copy full text from last result
            if self._last_result:
                self.clipboard_append(self._last_result.get('text', ''))
            else:
                self.clipboard_append(self.summary_tab.get("1.0", tk.END))

    def run(self):
        raise NotImplementedError

    def _toggle(self, widget, show, **pack_kwargs):
        if show:
            widget.pack(**pack_kwargs)
        else:
            widget.pack_forget()

    # --- Widget helpers ---
    def add_text_input(self, parent, label, hint="", height=3, width=50):
        fr = tk.Frame(parent, bg=MAIN_BG)
        fr.pack(fill=tk.X, pady=2)
        ttk.Label(fr, text=label).pack(anchor="w")
        txt = tk.Text(fr, height=height, width=width, font=MONO_FONT, relief=tk.SOLID, bd=1)
        txt.pack(fill=tk.X)
        if hint:
            ttk.Label(fr, text=hint, foreground="#888", font=(FONT_FAMILY, 8)).pack(anchor="w")
        return txt

    def add_entry(self, parent, label, default="", width=12):
        fr = tk.Frame(parent, bg=MAIN_BG)
        fr.pack(side=tk.LEFT, padx=(0, 10), pady=2)
        ttk.Label(fr, text=label).pack(anchor="w")
        var = tk.StringVar(value=default)
        ent = ttk.Entry(fr, textvariable=var, width=width)
        ent.pack()
        var._frame = fr
        return var

    def add_combo(self, parent, label, values, default=None, width=12):
        fr = tk.Frame(parent, bg=MAIN_BG)
        fr.pack(side=tk.LEFT, padx=(0, 10), pady=2)
        ttk.Label(fr, text=label).pack(anchor="w")
        var = tk.StringVar(value=default or values[0])
        cb = ttk.Combobox(fr, textvariable=var, values=values, width=width, state="readonly")
        cb.pack()
        var._frame = fr
        var._combobox = cb
        return var

    def add_button_group(self, parent, label, values, default=None, on_change=None):
        """A row of toggle buttons for selecting one value, like radio buttons."""
        fr = tk.Frame(parent, bg=MAIN_BG)
        fr.pack(side=tk.LEFT, padx=(0, 10), pady=2)
        ttk.Label(fr, text=label).pack(anchor="w")
        btn_row = tk.Frame(fr, bg=MAIN_BG)
        btn_row.pack(anchor="w")
        var = tk.StringVar(value=default or values[0])
        btns = {}

        def _select(val):
            var.set(val)
            _update()
            if on_change:
                on_change()

        def _update():
            cur = var.get()
            for v, b in btns.items():
                if v == cur:
                    b._bg_color = ACCENT
                    b._fg_color = "#fff"
                    b._hover_color = "#3a3d5c"
                    b._draw(ACCENT)
                else:
                    b._bg_color = "#e0e0e0"
                    b._fg_color = "#222"
                    b._hover_color = "#c8c8c8"
                    b._draw("#e0e0e0")

        for i, val in enumerate(values):
            b = RoundedButton(btn_row, text=val, command=lambda v=val: _select(v),
                              font=(FONT_FAMILY, 10), padx=10, pady=2,
                              bg_color="#e0e0e0", fg_color="#222",
                              hover_color="#c8c8c8", press_color="#b0b0b0")
            b.pack(side=tk.LEFT, padx=(0, 3))
            btns[val] = b

        _update()
        var._frame = fr
        var._btns = btns
        var._update = _update
        return var

    def add_check(self, parent, label, default=False):
        fr = tk.Frame(parent, bg=MAIN_BG)
        fr.pack(side=tk.LEFT, padx=(0, 10), pady=2)
        var = tk.BooleanVar(value=default)
        ToggleSwitch(fr, variable=var, text=label).pack()
        var._frame = fr
        return var

    def get_text(self, widget):
        return widget.get("1.0", tk.END).strip()

    def add_matrix_grid(self, parent, label, rows=2, cols=2, vector_mode=False,
                        row_label="samples", col_label="features", on_resize=None,
                        hide_rows=False, hide_cols=False):
        grid = MatrixGrid(parent, label, rows, cols, vector_mode, row_label, col_label,
                          on_resize=on_resize, hide_rows=hide_rows, hide_cols=hide_cols)
        grid.pack(fill=tk.X, pady=2)
        return grid


# ============================================================================
# MODULE: Classification
# ============================================================================

class ClassificationFrame(ModuleFrame):
    def __init__(self, parent):
        super().__init__(parent, "Classification (Logistic Regression)")

        f = self.input_frame
        self.X_grid = self.add_matrix_grid(f, "X (feature matrix)", rows=3, cols=2,
                                           row_label="samples", col_label="features")

        # Mode toggle buttons (Train / Predict)
        mode_frame = tk.Frame(f, bg=MAIN_BG)
        mode_frame.pack(fill=tk.X, pady=(4, 4))
        self._mode_val = "train"
        self._train_btn = RoundedButton(mode_frame, text="Train", command=lambda: self._set_mode("train"),
                                        font=(FONT_FAMILY, 11, "bold"), padx=16, pady=3,
                                        bg_color=ACCENT, fg_color="#fff",
                                        hover_color="#3a3d5c", press_color="#1a1d32")
        self._train_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._predict_btn = RoundedButton(mode_frame, text="Predict", command=lambda: self._set_mode("predict"),
                                          font=(FONT_FAMILY, 11, "bold"), padx=16, pady=3,
                                          bg_color="#e0e0e0", fg_color="#222",
                                          hover_color="#c8c8c8", press_color="#b0b0b0")
        self._predict_btn.pack(side=tk.LEFT)

        # --- Train widgets ---
        self.y_grid = self.add_matrix_grid(f, "y (labels)", rows=3, cols=1,
                                           row_label="samples", col_label="classes",
                                           hide_rows=True)
        self._x_rows_syncing = False
        self.X_grid.rows_var.trace_add("write", self._on_X_rows_var_change)

        self._row1 = tk.Frame(f, bg=MAIN_BG)
        self._row1.pack(fill=tk.X, pady=4)
        self.lr_var = self.add_entry(self._row1, "Learning Rate", "0.1")
        self.iters_var = self.add_entry(self._row1, "Epochs", "300")
        self.l2_var = self.add_entry(self._row1, "L2 Lambda", "0.0", width=6)
        self.threshold_var = self.add_entry(self._row1, "Threshold", "0.5", width=6)

        self._row2 = tk.Frame(f, bg=MAIN_BG)
        self._row2.pack(fill=tk.X, pady=4)
        self.pen_bias_var = self.add_check(self._row2, "Penalize Bias")
        self.metrics_var = self.add_check(self._row2, "Show Metrics", True)

        # Extra settings (batch, batch size, momentum) - stored as StringVars with defaults
        self.batch_var = tk.StringVar(value="gd")
        self.bs_var = tk.StringVar(value="0")
        self.momentum_var = tk.StringVar(value="0.0")

        self._extra_btn_frame = tk.Frame(f, bg=MAIN_BG)
        self._extra_btn_frame.pack(fill=tk.X, pady=(2, 4))
        self._extra_btn = RoundedButton(self._extra_btn_frame, text="Extra Settings",
                      command=self._open_extra_settings,
                      font=(FONT_FAMILY, 10), padx=10, pady=2,
                      bg_color="#e0e0e0", hover_color="#c8c8c8", press_color="#b0b0b0")
        self._extra_btn.pack(anchor="w")

        self._row3 = tk.Frame(f, bg=MAIN_BG)
        self._row3.pack(fill=tk.X, pady=4)
        self.w_init_var = self.add_button_group(self._row3, "W Init", ["zeros", "manual", "random"], "zeros",
                                                on_change=self._on_mode_change)
        self._train_w_grid = self.add_matrix_grid(f, "W (manual init)", rows=3, cols=1,
                                                   row_label="weights", col_label="classes")

        # --- Predict widgets ---
        self._pred_frame = tk.Frame(f, bg=MAIN_BG)
        self._pred_frame.pack(fill=tk.X, pady=4)
        self.pred_binary_var = self.add_check(self._pred_frame, "Binary")
        self.pred_threshold_var = self.add_entry(self._pred_frame, "Threshold", "0.5", width=6)

        self._pred_w_grid = self.add_matrix_grid(f, "W (trained weights)", rows=3, cols=1,
                                                  row_label="weights", col_label="classes")

        # "Use for Predict" button in button bar
        self._use_predict_btn = RoundedButton(self._btn_frame, text="Use for Predict",
                      command=self._use_for_predict,
                      font=(FONT_FAMILY, 10), padx=10, pady=2,
                      bg_color="#a6e3a1", fg_color="#1e1e2e",
                      hover_color="#94d990", press_color="#7cc97a")
        self._use_predict_btn.pack(side=tk.LEFT, padx=(8, 0))

        self._set_mode("train")

    def _use_for_predict(self):
        if self._last_result is None or self._last_result.get('weights') is None:
            return
        W = self._last_result['weights']
        self._set_mode("predict")
        self._pred_w_grid.set_from_matrix(np.atleast_2d(W))

    def _on_X_rows_var_change(self, *args):
        if self._x_rows_syncing:
            return
        try:
            new_rows = int(self.X_grid.rows_var.get())
        except (ValueError, tk.TclError):
            return
        new_rows = max(1, min(50, new_rows))
        if new_rows != self.y_grid.n_rows:
            self._x_rows_syncing = True
            self.y_grid._resize(new_rows, self.y_grid.n_cols)
            self._x_rows_syncing = False

    def _open_extra_settings(self):
        win = tk.Toplevel(self)
        win.title("Extra Settings – Classification")
        win.configure(bg=MAIN_BG)
        win.resizable(False, False)
        win.grab_set()
        # Position near the button
        bx = self._extra_btn.winfo_rootx()
        by = self._extra_btn.winfo_rooty() + self._extra_btn.winfo_height()
        win.geometry(f"+{bx}+{by}")

        pad = {'padx': 10, 'pady': 6}

        tk.Label(win, text="Batch Type", bg=MAIN_BG, font=(FONT_FAMILY, 11)).grid(row=0, column=0, sticky="w", **pad)
        batch_cb = ttk.Combobox(win, values=["gd", "sgd", "mb"], state="readonly", width=8)
        batch_cb.set(self.batch_var.get())
        batch_cb.grid(row=0, column=1, sticky="w", **pad)

        tk.Label(win, text="Batch Size", bg=MAIN_BG, font=(FONT_FAMILY, 11)).grid(row=1, column=0, sticky="w", **pad)
        bs_ent = tk.Entry(win, width=8, font=(FONT_FAMILY, 11))
        bs_ent.insert(0, self.bs_var.get())
        bs_ent.grid(row=1, column=1, sticky="w", **pad)

        tk.Label(win, text="Momentum", bg=MAIN_BG, font=(FONT_FAMILY, 11)).grid(row=2, column=0, sticky="w", **pad)
        mom_ent = tk.Entry(win, width=8, font=(FONT_FAMILY, 11))
        mom_ent.insert(0, self.momentum_var.get())
        mom_ent.grid(row=2, column=1, sticky="w", **pad)

        def _save():
            self.batch_var.set(batch_cb.get())
            self.bs_var.set(bs_ent.get())
            self.momentum_var.set(mom_ent.get())
            win.destroy()

        RoundedButton(win, text="OK", command=_save,
                      font=(FONT_FAMILY, 11, "bold"), padx=20, pady=4,
                      bg_color=ACCENT, fg_color="#fff",
                      hover_color="#3a3d5c", press_color="#1a1d32").grid(row=3, column=0, columnspan=2, pady=10)

    def _set_mode(self, mode):
        self._mode_val = mode
        # Update button styles
        if mode == "train":
            self._train_btn._bg_color = ACCENT
            self._train_btn._fg_color = "#fff"
            self._train_btn._hover_color = "#3a3d5c"
            self._train_btn._draw(ACCENT)
            self._predict_btn._bg_color = "#e0e0e0"
            self._predict_btn._fg_color = "#222"
            self._predict_btn._hover_color = "#c8c8c8"
            self._predict_btn._draw("#e0e0e0")
        else:
            self._predict_btn._bg_color = ACCENT
            self._predict_btn._fg_color = "#fff"
            self._predict_btn._hover_color = "#3a3d5c"
            self._predict_btn._draw(ACCENT)
            self._train_btn._bg_color = "#e0e0e0"
            self._train_btn._fg_color = "#222"
            self._train_btn._hover_color = "#c8c8c8"
            self._train_btn._draw("#e0e0e0")
        self._on_mode_change()

    def _on_mode_change(self):
        is_train = (self._mode_val == "train")
        w_init = self.w_init_var.get()

        # Train-only widgets
        self._toggle(self.y_grid, is_train, fill=tk.X, pady=2)
        self._toggle(self._row1, is_train, fill=tk.X, pady=4)
        self._toggle(self._row2, is_train, fill=tk.X, pady=4)
        self._toggle(self._extra_btn_frame, is_train, fill=tk.X, pady=(2, 4))
        self._toggle(self._row3, is_train, fill=tk.X, pady=4)
        self._toggle(self._train_w_grid, is_train and w_init == "manual", fill=tk.X, pady=2)

        # Predict-only widgets
        self._toggle(self._pred_frame, not is_train, fill=tk.X, pady=4)
        self._toggle(self._pred_w_grid, not is_train, fill=tk.X, pady=2)

    def run(self):
        try:
            if self._mode_val == "train":
                w_str = self._train_w_grid.get_matrix_string()
            else:
                w_str = self._pred_w_grid.get_matrix_string()
            result = compute_classification(
                X_str=self.X_grid.get_matrix_string(),
                y_str=self.y_grid.get_matrix_string(),
                mode_choice=self._mode_val,
                lr=float(self.lr_var.get()),
                iters=int(self.iters_var.get()),
                batch_type=self.batch_var.get(),
                batch_size=int(self.bs_var.get()),
                momentum=float(self.momentum_var.get()),
                l2=float(self.l2_var.get()),
                penalize_bias=self.pen_bias_var.get(),
                threshold=float(self.threshold_var.get()),
                w_init_choice=self.w_init_var.get(),
                w_init_str=w_str,
                pred_binary=self.pred_binary_var.get(),
                pred_W_str=w_str,
                pred_threshold=float(self.pred_threshold_var.get()),
                show_metrics=self.metrics_var.get(),
            )
            self.show_result(result)
        except Exception as e:
            self.show_output(f"ERROR: {e}")


# ============================================================================
# MODULE: Clustering
# ============================================================================

class ClusteringFrame(ModuleFrame):
    def __init__(self, parent):
        super().__init__(parent, "Clustering (K-Means / Fuzzy C-Means)")

        f = self.input_frame
        self.X_grid = self.add_matrix_grid(f, "X (data matrix)", rows=4, cols=1,
                                           row_label="samples", col_label="features")

        row1 = tk.Frame(f, bg=MAIN_BG)
        row1.pack(fill=tk.X, pady=4)
        self.method_var = self.add_button_group(row1, "Method", ["kmeans", "kmeans++", "fcm"], "kmeans",
                                                on_change=self._on_method_change)
        self.K_var = self.add_entry(row1, "K (for K++)", "2", width=6)
        self.fuzz_var = self.add_entry(row1, "Fuzzifier m", "2.0", width=6)

        self.C0_grid = self.add_matrix_grid(f, "C0 (initial centroids)", rows=2, cols=1,
                                            row_label="centroids", col_label="features",
                                            hide_cols=True)
        self._x_cols_syncing = False
        self.X_grid.cols_var.trace_add("write", self._on_X_cols_var_change)

        # Extra settings (Max Iter, Tol) - stored as StringVars with defaults
        self.maxiter_var = tk.StringVar(value="200")
        self.tol_var = tk.StringVar(value="1e-4")

        extra_btn_frame = tk.Frame(f, bg=MAIN_BG)
        extra_btn_frame.pack(fill=tk.X, pady=(2, 4))
        self._extra_btn = RoundedButton(extra_btn_frame, text="Extra Settings",
                      command=self._open_extra_settings,
                      font=(FONT_FAMILY, 10), padx=10, pady=2,
                      bg_color="#e0e0e0", hover_color="#c8c8c8", press_color="#b0b0b0")
        self._extra_btn.pack(anchor="w")

        self._on_method_change()

    def _on_X_cols_var_change(self, *args):
        if self._x_cols_syncing:
            return
        try:
            new_cols = int(self.X_grid.cols_var.get())
        except (ValueError, tk.TclError):
            return
        new_cols = max(1, min(50, new_cols))
        if new_cols != self.C0_grid.n_cols:
            self._x_cols_syncing = True
            self.C0_grid._resize(self.C0_grid.n_rows, new_cols)
            self._x_cols_syncing = False

    def _on_method_change(self):
        method = self.method_var.get()
        # kmeans: show C0, hide K, hide fuzz
        # kmeans++: hide C0, show K, hide fuzz
        # fcm: show C0, hide K, show fuzz
        self._toggle(self.C0_grid, method != "kmeans++", fill=tk.X, pady=2)
        self._toggle(self.K_var._frame, method == "kmeans++", side=tk.LEFT, padx=(0, 10), pady=2)
        self._toggle(self.fuzz_var._frame, method == "fcm", side=tk.LEFT, padx=(0, 10), pady=2)

    def _open_extra_settings(self):
        win = tk.Toplevel(self)
        win.title("Extra Settings – Clustering")
        win.configure(bg=MAIN_BG)
        win.resizable(False, False)
        win.grab_set()
        # Position near the button
        bx = self._extra_btn.winfo_rootx()
        by = self._extra_btn.winfo_rooty() + self._extra_btn.winfo_height()
        win.geometry(f"+{bx}+{by}")

        pad = {'padx': 10, 'pady': 6}

        tk.Label(win, text="Max Iter", bg=MAIN_BG, font=(FONT_FAMILY, 11)).grid(row=0, column=0, sticky="w", **pad)
        maxiter_ent = tk.Entry(win, width=8, font=(FONT_FAMILY, 11))
        maxiter_ent.insert(0, self.maxiter_var.get())
        maxiter_ent.grid(row=0, column=1, sticky="w", **pad)

        tk.Label(win, text="Tol", bg=MAIN_BG, font=(FONT_FAMILY, 11)).grid(row=1, column=0, sticky="w", **pad)
        tol_ent = tk.Entry(win, width=10, font=(FONT_FAMILY, 11))
        tol_ent.insert(0, self.tol_var.get())
        tol_ent.grid(row=1, column=1, sticky="w", **pad)

        def _save():
            self.maxiter_var.set(maxiter_ent.get())
            self.tol_var.set(tol_ent.get())
            win.destroy()

        RoundedButton(win, text="OK", command=_save,
                      font=(FONT_FAMILY, 11, "bold"), padx=20, pady=4,
                      bg_color=ACCENT, fg_color="#fff",
                      hover_color="#3a3d5c", press_color="#1a1d32").grid(row=2, column=0, columnspan=2, pady=10)

    def run(self):
        try:
            result = compute_clustering(
                X_str=self.X_grid.get_matrix_string(),
                method=self.method_var.get(),
                C0_str=self.C0_grid.get_matrix_string(),
                K_val=int(self.K_var.get()),
                fuzzifier=float(self.fuzz_var.get()),
                max_iter=int(self.maxiter_var.get()),
                tol=float(self.tol_var.get()),
            )
            self.show_result(result)
        except Exception as e:
            self.show_output(f"ERROR: {e}")


# ============================================================================
# MODULE: Gradient Descent
# ============================================================================

class GradientDescentFrame(ModuleFrame):
    def __init__(self, parent):
        super().__init__(parent, "Gradient Descent (Linear / Softmax)")

        f = self.input_frame
        self.X_grid = self.add_matrix_grid(f, "X (WITHOUT bias)", rows=3, cols=2,
                                           row_label="samples", col_label="features")
        self.y_grid = self.add_matrix_grid(f, "y / labels", rows=3, cols=1,
                                           row_label="samples", col_label="outputs",
                                           hide_rows=True)
        # Auto-sync y rows from X
        self._x_rows_syncing = False
        self.X_grid.rows_var.trace_add("write", self._on_X_rows_var_change)
        self.W0_grid = self.add_matrix_grid(f, "W0 (initial weights)", rows=3, cols=1,
                                            row_label="weights", col_label="outputs")

        row1 = tk.Frame(f, bg=MAIN_BG)
        row1.pack(fill=tk.X, pady=4)
        self.mode_var = self.add_button_group(row1, "Mode", ["linear", "softmax"], "linear",
                                               on_change=self._on_mode_change)
        self.lr_var = self.add_entry(row1, "lr", "0.1")
        self.iters_var = self.add_entry(row1, "Iterations", "10")
        self.nclass_var = self.add_entry(row1, "Num Classes", "3", width=6)

        row2 = tk.Frame(f, bg=MAIN_BG)
        row2.pack(fill=tk.X, pady=4)
        self.bias_var = self.add_check(row2, "Add Bias Column", True)
        self.norm_var = self.add_check(row2, "1/N Scaling", True)
        self.idx1_var = self.add_check(row2, "Labels 1-indexed", True)

        self._on_mode_change()

    def _on_X_rows_var_change(self, *args):
        if self._x_rows_syncing:
            return
        try:
            new_rows = int(self.X_grid.rows_var.get())
        except (ValueError, tk.TclError):
            return
        new_rows = max(1, min(50, new_rows))
        if new_rows != self.y_grid.n_rows:
            self._x_rows_syncing = True
            self.y_grid._resize(new_rows, self.y_grid.n_cols)
            self._x_rows_syncing = False

    def _on_mode_change(self):
        is_softmax = self.mode_var.get() == "softmax"
        self._toggle(self.nclass_var._frame, is_softmax, side=tk.LEFT, padx=(0, 10), pady=2)
        self._toggle(self.idx1_var._frame, is_softmax, side=tk.LEFT, padx=(0, 10), pady=2)

    def run(self):
        try:
            result = compute_gradient_descent(
                mode=self.mode_var.get(),
                X_str=self.X_grid.get_matrix_string(),
                y_str=self.y_grid.get_matrix_string(),
                W0_str=self.W0_grid.get_matrix_string(),
                lr=float(self.lr_var.get()),
                iters=int(self.iters_var.get()),
                add_bias_col=self.bias_var.get(),
                normalize=self.norm_var.get(),
                n_classes=int(self.nclass_var.get()),
                labels_1indexed=self.idx1_var.get(),
            )
            self.show_result(result)
        except Exception as e:
            self.show_output(f"ERROR: {e}")


# ============================================================================
# MODULE: Neural Network
# ============================================================================

class NeuralNetFrame(ModuleFrame):
    def __init__(self, parent):
        super().__init__(parent, "Neural Network (MLP)")

        f = self.input_frame
        self.X_grid = self.add_matrix_grid(f, "X (NO bias)", rows=3, cols=2,
                                           row_label="samples", col_label="features")
        self.Y_grid = self.add_matrix_grid(f, "Y (targets)", rows=3, cols=2,
                                           row_label="samples", col_label="outputs",
                                           hide_rows=True)
        # Auto-sync Y rows from X
        self._x_rows_syncing = False
        self.X_grid.rows_var.trace_add("write", self._on_X_rows_var_change)

        # Layer controls
        layer_header = tk.Frame(f, bg=MAIN_BG)
        layer_header.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(layer_header, text="Layers", style="Header.TLabel").pack(side=tk.LEFT)
        RoundedButton(layer_header, text="+ Add Layer", command=self.add_layer
                      ).pack(side=tk.LEFT, padx=8)

        self.layers_frame = tk.Frame(f, bg=MAIN_BG)
        self.layers_frame.pack(fill=tk.X)
        self.layer_widgets = []
        self.add_layer()  # default 1 layer

        # Auto-sync W grid dimensions when X cols change
        self.X_grid.cols_var.trace_add("write", lambda *a: self._sync_all_W_dims())

        # Hyperparams
        row = tk.Frame(f, bg=MAIN_BG)
        row.pack(fill=tk.X, pady=8)
        self.lr_var = self.add_entry(row, "lr", "0.1")
        self.iters_var = self.add_entry(row, "Iterations", "1")
        self.loss_var = self.add_combo(row, "Loss", ["mse", "ce"], "mse")

    def add_layer(self):
        idx = len(self.layer_widgets)
        fr = tk.LabelFrame(self.layers_frame, text=f"Layer {idx + 1}", bg=MAIN_BG,
                           font=(FONT_FAMILY, 9, "bold"), padx=6, pady=4)
        fr.pack(fill=tk.X, pady=2)

        row = tk.Frame(fr, bg=MAIN_BG)
        row.pack(fill=tk.X)

        ttk.Label(row, text="Neurons:").pack(side=tk.LEFT)
        neurons_var = tk.StringVar(value="2")
        ttk.Entry(row, textvariable=neurons_var, width=5).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(row, text="Activation:").pack(side=tk.LEFT)
        act_var = tk.StringVar(value="relu")
        ttk.Combobox(row, textvariable=act_var, values=["relu", "sigmoid", "linear", "softmax", "tanh"],
                     width=8, state="readonly").pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(row, text="Init:").pack(side=tk.LEFT)
        init_var = tk.StringVar(value="xavier")
        ttk.Combobox(row, textvariable=init_var, values=["zeros", "xavier", "he", "random", "manual"],
                     width=8, state="readonly").pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(row, text="Seed:").pack(side=tk.LEFT)
        seed_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=seed_var, width=5).pack(side=tk.LEFT, padx=(2, 10))

        # Manual W input (rows and cols auto-synced)
        w_grid = MatrixGrid(fr, "Manual W (incl bias row)", rows=3, cols=2,
                            row_label="inputs+bias", col_label="neurons",
                            hide_rows=True, hide_cols=True)
        w_grid.pack(fill=tk.X, pady=2)

        # Remove button at the bottom of this layer
        remove_btn = RoundedButton(fr, text="Remove this layer",
                                   command=lambda: self.remove_layer(lw),
                                   font=(FONT_FAMILY, 9), bg_color="#e8c0c0",
                                   hover_color="#d9a0a0", press_color="#c88080")
        remove_btn.pack(anchor="e", pady=(2, 0))

        lw = {
            'frame': fr, 'neurons': neurons_var, 'activation': act_var,
            'init': init_var, 'seed': seed_var, 'W_manual': w_grid
        }
        self.layer_widgets.append(lw)

        # Sync W cols when neurons changes, and sync all W rows (chain effect)
        neurons_var.trace_add("write", lambda *args, l=lw: self._on_neurons_change(l))
        init_var.trace_add("write", lambda *args, l=lw: self._on_init_change(l))
        self._sync_all_W_dims()
        self._on_init_change(lw)

    def _on_X_rows_var_change(self, *args):
        if self._x_rows_syncing:
            return
        try:
            new_rows = int(self.X_grid.rows_var.get())
        except (ValueError, tk.TclError):
            return
        new_rows = max(1, min(50, new_rows))
        if new_rows != self.Y_grid.n_rows:
            self._x_rows_syncing = True
            self.Y_grid._resize(new_rows, self.Y_grid.n_cols)
            self._x_rows_syncing = False

    def _on_neurons_change(self, lw):
        """When a layer's neuron count changes, sync its W cols and re-sync all W rows."""
        if lw not in self.layer_widgets:
            return
        try:
            n = int(lw['neurons'].get())
        except (ValueError, tk.TclError):
            return
        n = max(1, min(50, n))
        w_grid = lw['W_manual']
        if w_grid.n_cols != n:
            w_grid._resize(w_grid.n_rows, n)
        # Changing neurons affects the NEXT layer's W rows
        self._sync_all_W_dims()

    def _sync_all_W_dims(self):
        """Sync W grid rows for all layers based on the chain: X cols -> layer neurons."""
        try:
            prev_dim = int(self.X_grid.cols_var.get())
        except (ValueError, tk.TclError):
            return
        for lw in self.layer_widgets:
            w_grid = lw['W_manual']
            expected_rows = prev_dim + 1  # +1 for bias
            try:
                n = int(lw['neurons'].get())
            except (ValueError, tk.TclError):
                n = w_grid.n_cols
            n = max(1, min(50, n))
            # Sync cols to neurons
            if w_grid.n_cols != n:
                w_grid._resize(w_grid.n_rows, n)
            # Sync rows to prev_dim + 1
            if w_grid.n_rows != expected_rows:
                w_grid._resize(expected_rows, w_grid.n_cols)
            prev_dim = n

    def _on_init_change(self, lw):
        if lw in self.layer_widgets:
            is_manual = lw['init'].get() == "manual"
            self._toggle(lw['W_manual'], is_manual, fill=tk.X, pady=2)

    def remove_layer(self, lw):
        if len(self.layer_widgets) <= 1:
            return
        self.layer_widgets.remove(lw)
        lw['frame'].destroy()
        # Renumber remaining layers
        for i, layer in enumerate(self.layer_widgets):
            layer['frame'].configure(text=f"Layer {i + 1}")
        # Re-sync W dimensions after removal
        self._sync_all_W_dims()

    def run(self):
        try:
            layer_configs = []
            for lw in self.layer_widgets:
                w_str = lw['W_manual'].get_matrix_string()
                # Check if all cells are just "0" (no manual input)
                all_zero = all(v.strip() in ("0", "0.0", "") for v in w_str.replace(",", " ").split())
                layer_configs.append({
                    'neurons': int(lw['neurons'].get()),
                    'activation': lw['activation'].get(),
                    'init': lw['init'].get(),
                    'seed': lw['seed'].get(),
                    'W_manual': w_str if not all_zero else None,
                })

            result = compute_neural(
                X_str=self.X_grid.get_matrix_string(),
                Y_str=self.Y_grid.get_matrix_string(),
                layer_configs=layer_configs,
                lr=float(self.lr_var.get()),
                iters=int(self.iters_var.get()),
                loss_type=self.loss_var.get(),
            )
            self.show_result(result)
        except Exception as e:
            self.show_output(f"ERROR: {e}")


# ============================================================================
# MODULE: Regression
# ============================================================================

class RegressionFrame(ModuleFrame):
    def __init__(self, parent):
        super().__init__(parent, "Regression (OLS / Ridge / Polynomial)")

        f = self.input_frame
        self.X_grid = self.add_matrix_grid(f, "X", rows=5, cols=1,
                                           row_label="samples", col_label="features")

        # Mode toggle buttons (Train / Predict)
        mode_frame = tk.Frame(f, bg=MAIN_BG)
        mode_frame.pack(fill=tk.X, pady=(4, 4))
        self._mode_val = "train"
        self._train_btn = RoundedButton(mode_frame, text="Train", command=lambda: self._set_mode("train"),
                                        font=(FONT_FAMILY, 11, "bold"), padx=16, pady=3,
                                        bg_color=ACCENT, fg_color="#fff",
                                        hover_color="#3a3d5c", press_color="#1a1d32")
        self._train_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._predict_btn = RoundedButton(mode_frame, text="Predict", command=lambda: self._set_mode("predict"),
                                          font=(FONT_FAMILY, 11, "bold"), padx=16, pady=3,
                                          bg_color="#e0e0e0", fg_color="#222",
                                          hover_color="#c8c8c8", press_color="#b0b0b0")
        self._predict_btn.pack(side=tk.LEFT)

        # --- Train widgets ---
        self.Y_grid = self.add_matrix_grid(f, "Y", rows=5, cols=1,
                                           row_label="samples", col_label="outputs",
                                           hide_rows=True)
        self._x_rows_syncing = False
        self.X_grid.rows_var.trace_add("write", self._on_X_rows_var_change)

        self._train_row = tk.Frame(f, bg=MAIN_BG)
        self._train_row.pack(fill=tk.X, pady=4)
        self.model_var = self.add_button_group(self._train_row, "Model", ["ols", "ridge", "polynomial"], "ols",
                                                on_change=self._on_mode_change)
        self.alpha_var = self.add_entry(self._train_row, "Alpha (Ridge)", "1.0")
        self.degree_var = self.add_entry(self._train_row, "Degree (Poly)", "2", width=6)
        self.pen_bias_var = self.add_check(self._train_row, "Penalize Bias")

        # --- Predict widgets ---
        self._pred_row = tk.Frame(f, bg=MAIN_BG)
        self._pred_row.pack(fill=tk.X, pady=4)
        self.pred_model_var = self.add_button_group(self._pred_row, "Model", ["ols/ridge", "polynomial"], "ols/ridge",
                                                    on_change=self._on_mode_change)
        self.pred_degree_var = self.add_entry(self._pred_row, "Degree (Poly)", "2", width=6)

        self.W_grid = self.add_matrix_grid(f, "W (trained weights)", rows=1, cols=1,
                                           row_label="features", col_label="outputs")
        self.b_grid = self.add_matrix_grid(f, "b (intercept)", rows=1, cols=1,
                                           row_label="row", col_label="outputs")

        # "Use for Predict" button in button bar
        self._use_predict_btn = RoundedButton(self._btn_frame, text="Use for Predict",
                      command=self._use_for_predict,
                      font=(FONT_FAMILY, 10), padx=10, pady=2,
                      bg_color="#a6e3a1", fg_color="#1e1e2e",
                      hover_color="#94d990", press_color="#7cc97a")
        self._use_predict_btn.pack(side=tk.LEFT, padx=(8, 0))

        self._set_mode("train")

    def _use_for_predict(self):
        if self._last_result is None or self._last_result.get('weights') is None:
            return
        W = self._last_result['weights']
        b = self._last_result.get('bias')
        self._set_mode("predict")
        self.W_grid.set_from_matrix(np.atleast_2d(W))
        if b is not None:
            self.b_grid.set_from_matrix(np.atleast_2d(np.atleast_1d(b)))

    def _on_X_rows_var_change(self, *args):
        if self._x_rows_syncing:
            return
        try:
            new_rows = int(self.X_grid.rows_var.get())
        except (ValueError, tk.TclError):
            return
        new_rows = max(1, min(50, new_rows))
        if new_rows != self.Y_grid.n_rows:
            self._x_rows_syncing = True
            self.Y_grid._resize(new_rows, self.Y_grid.n_cols)
            self._x_rows_syncing = False

    def _set_mode(self, mode):
        self._mode_val = mode
        if mode == "train":
            self._train_btn._bg_color = ACCENT
            self._train_btn._fg_color = "#fff"
            self._train_btn._hover_color = "#3a3d5c"
            self._train_btn._draw(ACCENT)
            self._predict_btn._bg_color = "#e0e0e0"
            self._predict_btn._fg_color = "#222"
            self._predict_btn._hover_color = "#c8c8c8"
            self._predict_btn._draw("#e0e0e0")
        else:
            self._predict_btn._bg_color = ACCENT
            self._predict_btn._fg_color = "#fff"
            self._predict_btn._hover_color = "#3a3d5c"
            self._predict_btn._draw(ACCENT)
            self._train_btn._bg_color = "#e0e0e0"
            self._train_btn._fg_color = "#222"
            self._train_btn._hover_color = "#c8c8c8"
            self._train_btn._draw("#e0e0e0")
        self._on_mode_change()

    def _on_mode_change(self):
        is_train = (self._mode_val == "train")
        model = self.model_var.get()
        pred_model = self.pred_model_var.get()

        # Train widgets
        self._toggle(self.Y_grid, is_train, fill=tk.X, pady=2)
        self._toggle(self._train_row, is_train, fill=tk.X, pady=4)
        if is_train:
            self._toggle(self.alpha_var._frame, model != "ols", side=tk.LEFT, padx=(0, 10), pady=2)
            self._toggle(self.degree_var._frame, model == "polynomial", side=tk.LEFT, padx=(0, 10), pady=2)
            self._toggle(self.pen_bias_var._frame, model != "ols", side=tk.LEFT, padx=(0, 10), pady=2)

        # Predict widgets
        self._toggle(self._pred_row, not is_train, fill=tk.X, pady=4)
        self._toggle(self.W_grid, not is_train, fill=tk.X, pady=2)
        self._toggle(self.b_grid, not is_train, fill=tk.X, pady=2)
        if not is_train:
            self._toggle(self.pred_degree_var._frame, pred_model == "polynomial",
                         side=tk.LEFT, padx=(0, 10), pady=2)

    def run(self):
        try:
            if self._mode_val == "train":
                result = compute_regression(
                    X_str=self.X_grid.get_matrix_string(),
                    Y_str=self.Y_grid.get_matrix_string(),
                    model=self.model_var.get(),
                    alpha=float(self.alpha_var.get()),
                    degree=int(self.degree_var.get()),
                    penalize_bias=self.pen_bias_var.get(),
                )
            else:
                model = "polynomial" if self.pred_model_var.get() == "polynomial" else "ols"
                result = compute_regression_predict(
                    X_str=self.X_grid.get_matrix_string(),
                    W_str=self.W_grid.get_matrix_string(),
                    b_str=self.b_grid.get_matrix_string(),
                    model=model,
                    degree=int(self.pred_degree_var.get()),
                )
            self.show_result(result)
        except Exception as e:
            self.show_output(f"ERROR: {e}")


# ============================================================================
# MODULE: Decision Tree
# ============================================================================

class DecisionTreeFrame(ModuleFrame):
    def __init__(self, parent):
        super().__init__(parent, "Decision Tree & Random Forest")

        f = self.input_frame

        # X grid — y sync is set up after y_grid is created
        self.X_grid = self.add_matrix_grid(f, "X", rows=5, cols=1,
                                           row_label="samples", col_label="features")
        self.y_grid = self.add_matrix_grid(f, "y", rows=5, cols=1,
                                           row_label="samples", col_label="outputs",
                                           hide_rows=True)
        # Wire up X → y row sync via variable trace (covers spinbox arrows + typed edits + paste)
        self._x_rows_syncing = False
        self.X_grid.rows_var.trace_add("write", self._on_X_rows_var_change)

        row1 = tk.Frame(f, bg=MAIN_BG)
        row1.pack(fill=tk.X, pady=4)
        self.task_var = self.add_combo(row1, "Task", ["classification", "regression"], "regression")
        self.crit_var = self.add_combo(row1, "Criterion", ["gini", "entropy"], "gini")
        self.mode_var = self.add_combo(row1, "Mode", ["root", "tree", "forest"], "root")

        row2 = tk.Frame(f, bg=MAIN_BG)
        row2.pack(fill=tk.X, pady=4)
        self.depth_var = self.add_entry(row2, "Max Depth", "3", width=6)
        self.mins_var = self.add_entry(row2, "Min Samples", "2", width=6)
        self.ntrees_var = self.add_entry(row2, "Num Trees (RF)", "10", width=6)
        self.feat_var = self.add_combo(row2, "Max Features", ["sqrt", "log2", "all"], "sqrt")

        self.thr_grid = self.add_matrix_grid(f, "Thresholds (optional, leave 0 = auto)",
                                             rows=1, cols=2,
                                             row_label="features", col_label="thresholds",
                                             hide_rows=True)
        # Auto-sync threshold rows from X cols (features)
        self.X_grid.cols_var.trace_add("write", self._on_X_cols_var_change)

        self.task_var._combobox.bind("<<ComboboxSelected>>", lambda e: self._on_mode_change())
        self.mode_var._combobox.bind("<<ComboboxSelected>>", lambda e: self._on_mode_change())
        self._on_mode_change()

    def _on_X_rows_var_change(self, *args):
        """Trace callback: sync y rows when X rows change (spinbox, paste, etc.)."""
        if self._x_rows_syncing:
            return
        try:
            new_rows = int(self.X_grid.rows_var.get())
        except (ValueError, tk.TclError):
            return
        new_rows = max(1, min(50, new_rows))
        if new_rows != self.y_grid.n_rows:
            self._x_rows_syncing = True
            self.y_grid._resize(new_rows, self.y_grid.n_cols)
            self._x_rows_syncing = False

    def _on_X_cols_var_change(self, *args):
        """Auto-sync threshold grid rows to match number of X features (columns)."""
        try:
            n_features = int(self.X_grid.cols_var.get())
        except (ValueError, tk.TclError):
            return
        n_features = max(1, min(50, n_features))
        if n_features != self.thr_grid.n_rows:
            self.thr_grid._resize(n_features, self.thr_grid.n_cols)

    def _thr_grid_to_str(self):
        """Convert threshold grid to the string format expected by parse_thresholds.
        Each row = a feature index, columns = candidate thresholds.
        Skips rows that are all zeros or empty. Returns format: '0: v1 v2, 1: v3'
        """
        parts = []
        for r in range(self.thr_grid.n_rows):
            vals = []
            for c in range(self.thr_grid.n_cols):
                v = self.thr_grid.cells[r][c].get().strip()
                if v and v != "0":
                    vals.append(v)
            if vals:
                parts.append(f"{r}: " + " ".join(vals))
        return ", ".join(parts)

    def _on_mode_change(self):
        task = self.task_var.get()
        mode = self.mode_var.get()

        # Criterion only for classification
        self._toggle(self.crit_var._frame, task == "classification", side=tk.LEFT, padx=(0, 10), pady=2)

        # root: hide depth, mins, ntrees, feat
        # tree: hide ntrees
        # forest: show all
        self._toggle(self.depth_var._frame, mode != "root", side=tk.LEFT, padx=(0, 10), pady=2)
        self._toggle(self.mins_var._frame, mode != "root", side=tk.LEFT, padx=(0, 10), pady=2)
        self._toggle(self.ntrees_var._frame, mode == "forest", side=tk.LEFT, padx=(0, 10), pady=2)
        self._toggle(self.feat_var._frame, mode == "forest", side=tk.LEFT, padx=(0, 10), pady=2)

    def run(self):
        try:
            result = compute_tree(
                X_str=self.X_grid.get_matrix_string(),
                y_str=self.y_grid.get_matrix_string(),
                task=self.task_var.get(),
                criterion=self.crit_var.get(),
                thr_str=self._thr_grid_to_str(),
                tree_mode=self.mode_var.get(),
                depth=int(self.depth_var.get()),
                min_samples=int(self.mins_var.get()),
                n_trees=int(self.ntrees_var.get()),
                feat_mode=self.feat_var.get(),
            )
            self.show_result(result)
        except Exception as e:
            self.show_output(f"ERROR: {e}")


# ============================================================================
# MODULE: Cost Minimizer
# ============================================================================

class CostMinimizerFrame(ModuleFrame):
    VAR_NAMES = ['x1', 'x2', 'x3', 'x4', 'x5']

    def __init__(self, parent):
        super().__init__(parent, "Cost Function Minimizer")

        f = self.input_frame

        # Expression editor (with embedded variable chips)
        expr_fr = tk.Frame(f, bg=MAIN_BG)
        expr_fr.pack(fill=tk.X, pady=2)
        ttk.Label(expr_fr, text="C(...) expression").pack(anchor="w")

        self.expr_editor = ExpressionEditor(expr_fr)
        self.expr_editor.pack(fill=tk.X)
        self.expr_editor.set_on_change(self._on_expr_change)

        ttk.Label(expr_fr,
                  text="e.g. sin(x\u2081)^2 + x\u2082^2  |  Use ^ for power. Insert variables with the blue buttons.",
                  foreground="#888", font=(FONT_FAMILY, 8)).pack(anchor="w")

        # Variable buttons row (blue) — only way to insert variables
        var_row = tk.Frame(f, bg=MAIN_BG)
        var_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(var_row, text="Variables:", font=(FONT_FAMILY, 9)).pack(side=tk.LEFT, padx=(0, 6))
        for name in self.VAR_NAMES:
            display = 'x' + name[1:].translate(_SUBSCRIPTS)
            RoundedButton(var_row, text=display,
                          command=lambda n=name: self.expr_editor.insert_variable(n),
                          font=(FONT_FAMILY, 10, "bold"), padx=10, pady=2,
                          bg_color=ACCENT, fg_color="#fff",
                          hover_color="#3a3d5c", press_color="#1a1d32"
                          ).pack(side=tk.LEFT, padx=(0, 4))

        # Math keyboard (operators & functions only — no variables)
        self.math_kb = MathKeyboard(f, self.expr_editor)
        self.math_kb.pack(fill=tk.X, pady=(4, 4))

        row1 = tk.Frame(f, bg=MAIN_BG)
        row1.pack(fill=tk.X, pady=4)
        self.lr_var = self.add_entry(row1, "lr", "0.1")
        self.iters_var = self.add_entry(row1, "Iterations", "1")

        # Initial value entries (shown per detected variable)
        self._init_label = ttk.Label(f, text="Initial values:",
                  foreground="#555", font=(FONT_FAMILY, 9))

        self._init_row = tk.Frame(f, bg=MAIN_BG)

        self._var_init = {}
        self._var_init_frames = {}

        for name in self.VAR_NAMES:
            fr = tk.Frame(self._init_row, bg=MAIN_BG)
            display = 'x' + name[1:].translate(_SUBSCRIPTS) + '(0)'
            ttk.Label(fr, text=display).pack(anchor="w")
            sv = tk.StringVar(value="0.0")
            ttk.Entry(fr, textvariable=sv, width=8).pack()
            self._var_init[name] = sv
            self._var_init_frames[name] = fr

        self._prev_vars = []
        self._on_expr_change()

    def _on_expr_change(self):
        detected = self.expr_editor.get_variables()
        if detected == self._prev_vars:
            return
        self._prev_vars = detected

        for name in self.VAR_NAMES:
            if name in detected:
                self._var_init_frames[name].pack(side=tk.LEFT, padx=(0, 10), pady=2)
            else:
                self._var_init_frames[name].pack_forget()

        has_vars = len(detected) > 0
        self._toggle(self._init_label, has_vars, anchor="w", pady=(6, 0))
        self._toggle(self._init_row, has_vars, fill=tk.X, pady=2)

    def run(self):
        try:
            expr = self.expr_editor.get_expression().strip()
            var_names = self.expr_editor.get_variables()
            if not var_names:
                self.show_output("ERROR: No variables in expression. Use the blue buttons to insert x\u2081\u2013x\u2085.")
                return
            init_vals = [float(self._var_init[n].get()) for n in var_names]

            result = compute_cost_minimizer(
                mode="custom", expr=expr, var_names=var_names,
                init_vals=init_vals, lr=float(self.lr_var.get()),
                iters=int(self.iters_var.get()),
            )
            self.show_result(result)
        except Exception as e:
            self.show_output(f"ERROR: {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    app = App()
    app.mainloop()
