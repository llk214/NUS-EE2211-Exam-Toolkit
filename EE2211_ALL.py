
import numpy as np
from itertools import combinations_with_replacement
import math
import re

try:
    print("""
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551                     EE2211 EXAM TOOLKIT (All-in-One)                     \u2551
\u2551  Logistic | Clustering | Regression | Neural Net | Trees | Optimizer     \u2551
\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d
""")
except UnicodeEncodeError:
    print("""
+--------------------------------------------------------------------------+
|                     EE2211 EXAM TOOLKIT (All-in-One)                     |
|  Logistic | Clustering | Regression | Neural Net | Trees | Optimizer     |
+--------------------------------------------------------------------------+
""")


# ============================================================================
# PROGRAM 1: CLASSIFICATION (Logistic Regression)
# ============================================================================

def run_classification():
    print("\n" + "=" * 70)
    print(" CLASSIFICATION (Logistic Regression - configurable) ".center(70))
    print("=" * 70)

    def parse_matrix(rows_str: str):
        rows = [r.strip() for r in rows_str.split(',') if r.strip()]
        return np.array([list(map(float, r.split())) for r in rows], dtype=float)

    def parse_y(y_str: str):
        if ',' in y_str:
            Y = parse_matrix(y_str)
            K = Y.shape[1]
            if K == 2:
                y = np.argmax(Y, axis=1).astype(int)
                return 'binary', y, 2
            return 'multiclass', Y, K
        else:
            y = np.array(list(map(float, y_str.split())), dtype=float)
            y = y.astype(int)
            classes = np.unique(y)
            if set(classes.tolist()) <= {0, 1}:
                return 'binary', y, 2
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y_idx = np.array([class_to_idx[c] for c in y], dtype=int)
            K = len(classes)
            Y = np.zeros((len(y), K), dtype=float)
            Y[np.arange(len(y)), y_idx] = 1
            return 'multiclass', Y, K

    def add_bias(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

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

    # optimizer helpers: batch / sgd / minibatch with optional momentum
    def apply_optimizer(Xb, y_or_Y, W, grads, opt_state, opt_config):
        # opt_config: dict {type:'gd'/'sgd'/'mb', lr, momentum, batch_size, l2}
        lr = opt_config.get('lr', 0.1)
        momentum = opt_config.get('momentum', 0.0)
        l2 = opt_config.get('l2', 0.0)
        g = grads
        # add L2 gradient (do not penalize bias column if penalize_bias False)
        if l2 != 0:
            W_reg = W.copy()
            if opt_config.get('penalize_bias', False) is False:
                W_reg[0, :] = 0.0
            g = g + l2 * W_reg
        # momentum
        v = opt_state.get('v', np.zeros_like(W))
        v = momentum * v + lr * g
        opt_state['v'] = v
        W_new = W - v
        return W_new, opt_state

    # training loops for binary / multiclass
    def train_binary(X, y, W_init=None, config=None):
        config = config or {}
        Xb = add_bias(X)
        n, d1 = Xb.shape
        if W_init is None:
            W = np.zeros(d1)
        else:
            W = W_init.astype(float)
        iters = config.get('iters', 300)
        opt_config = dict(
            lr=config.get('lr', 0.1),
            momentum=config.get('momentum', 0.0),
            l2=config.get('l2', 0.0),
            penalize_bias=config.get('penalize_bias', False),
        )
        opt_state = {'v': np.zeros_like(W)}
        batch = config.get('batch', 'gd')
        bs = config.get('batch_size', n)
        loss_history = []
        for epoch in range(1, iters + 1):
            # shuffle if SGD/mini-batch
            idxs = np.arange(n)
            if batch != 'gd':
                np.random.shuffle(idxs)
            for start in range(0, n, bs):
                batch_idx = idxs[start:start+bs]
                Xb_batch = Xb[batch_idx]
                y_batch = y[batch_idx]
                p = sigmoid(Xb_batch @ W)
                grad = (Xb_batch.T @ (p - y_batch)) / len(y_batch)
                # shape align
                Wshape = (d1,)
                grad = grad.reshape(d1,)
                W_mat = W.reshape(-1)
                W_mat, opt_state = apply_optimizer(Xb_batch, y_batch, W_mat, grad, opt_state, opt_config)
                W = W_mat
            # logging
            p_all = sigmoid(Xb @ W)
            loss = bce_loss(y, p_all)
            loss_history.append(loss)
            if epoch % config.get('print_every', 20) == 0 or epoch == 1 or epoch == iters:
                acc = np.mean((p_all >= config.get('threshold', 0.5)).astype(int) == y)
                print(f"[BIN] epoch {epoch}/{iters} | loss={loss:.8f} acc={acc:.4f}")
            # early stopping on small change
            if len(loss_history) > 2 and abs(loss_history[-1] - loss_history[-2]) < config.get('tol', 1e-9):
                print("Early stop: loss change < tol")
                break
        return W

    def train_multiclass(X, Y, W_init=None, config=None):
        config = config or {}
        Xb = add_bias(X)
        n, d1 = Xb.shape
        K = Y.shape[1]
        if W_init is None:
            W = np.zeros((d1, K))
        else:
            W = W_init.astype(float)
        iters = config.get('iters', 300)
        opt_config = dict(
            lr=config.get('lr', 0.1),
            momentum=config.get('momentum', 0.0),
            l2=config.get('l2', 0.0),
            penalize_bias=config.get('penalize_bias', False),
        )
        opt_state = {'v': np.zeros_like(W)}
        batch = config.get('batch', 'gd')
        bs = config.get('batch_size', n)
        loss_history = []
        for epoch in range(1, iters + 1):
            idxs = np.arange(n)
            if batch != 'gd':
                np.random.shuffle(idxs)
            for start in range(0, n, bs):
                batch_idx = idxs[start:start+bs]
                Xb_batch = Xb[batch_idx]
                Y_batch = Y[batch_idx]
                P = softmax(Xb_batch @ W)
                grad = (Xb_batch.T @ (P - Y_batch)) / len(Y_batch)
                W, opt_state = apply_optimizer(Xb_batch, Y_batch, W, grad, opt_state, opt_config)
            P_all = softmax(Xb @ W)
            loss = cce_loss(Y, P_all)
            loss_history.append(loss)
            if epoch % config.get('print_every', 20) == 0 or epoch == 1 or epoch == iters:
                acc = np.mean(np.argmax(P_all, axis=1) == np.argmax(Y, axis=1))
                print(f"[MC ] epoch {epoch}/{iters} | loss={loss:.8f} acc={acc:.4f}")
            if len(loss_history) > 2 and abs(loss_history[-1] - loss_history[-2]) < config.get('tol', 1e-9):
                print("Early stop: loss change < tol")
                break
        return W

    # --- user input / choices ---
    X = parse_matrix(input("X rows (comma-separated) = ").strip())
    print("Do you have labels to train (t) or only want to predict with known W (p)?")
    mode_choice = input("(t/p): ").strip().lower() or "t"
    if mode_choice == 'p':
        is_binary = input("Binary or multiclass? (b/m): ").strip().lower() or 'b'
        if is_binary == 'b':
            W = np.array(list(map(float, input("W vector (bias first) = ").split())))
            Xb = add_bias(X)
            p = sigmoid(Xb @ W)
            th = float(input("Threshold for label (default 0.5): ").strip() or "0.5")
            print("Probabilities:", np.round(p, 8))
            print("Labels:", (p >= th).astype(int))
        else:
            W = parse_matrix(input("W matrix (rows=bias+features, cols=K) = ").strip())
            Xb = add_bias(X)
            P = softmax(Xb @ W)
            print("Probabilities:\n", np.round(P, 8))
            print("Labels:", np.argmax(P, axis=1))
        return

    mode, y_or_Y, K = parse_y(input("y = ").strip())
    # hyperparams
    lr = float(input("Learning rate (default 0.1): ").strip() or "0.1")
    iters = int(input("Epochs/iterations (default 300): ").strip() or "300")
    batch_type = input("Batch mode: 'gd' / 'sgd' / 'mb' (default 'gd'): ").strip() or "gd"
    bs = int(input("Batch size for minibatch (default whole data): ").strip() or str(X.shape[0]))
    momentum = float(input("Momentum (0 for none): ").strip() or "0.0")
    l2 = float(input("L2 regularization lambda (0 for none): ").strip() or "0.0")
    penalize_bias = input("Penalize bias in L2? (EE2211,EE2213 choose y)(y/n default n): ").strip().lower() == 'y'
    threshold = float(input("Binary threshold (default 0.5): ").strip() or "0.5")

    config = dict(lr=lr, iters=iters, batch=batch_type, batch_size=bs, momentum=momentum, l2=l2,
                  penalize_bias=penalize_bias, print_every=20, tol=1e-9, threshold=threshold)

    w_init_choice = input("Initial weights: 1=zeros 2=manual 3=random (normal) : ").strip() or "1"
    if mode == 'binary':
        if w_init_choice == '2':
            w0 = np.array(list(map(float, input(f"Enter vector (bias+{X.shape[1]}): ").split())))
        elif w_init_choice == '3':
            seed = input("Seed for random init (blank => random): ").strip()
            if seed: np.random.seed(int(seed))
            w0 = np.random.randn(X.shape[1] + 1)
        else:
            w0 = None
        W_tr = train_binary(X, y_or_Y, W_init=w0, config=config)
        print("Trained W:", np.round(W_tr, 8))
    else:
        # multiclass
        if w_init_choice == '2':
            W0 = parse_matrix(input(f"Enter initial W matrix (rows={X.shape[1]+1}, cols={K}): ").strip())
        elif w_init_choice == '3':
            seed = input("Seed for rand init (blank => random): ").strip()
            if seed: np.random.seed(int(seed))
            W0 = np.random.randn(X.shape[1] + 1, K)
        else:
            W0 = None
        W_tr = train_multiclass(X, y_or_Y, W_init=W0, config=config)
        print("Trained W:\n", np.round(W_tr, 8))

    # final eval printing: confusion + precision/recall/F1 (if binary)
    if input("Print confusion/metrics? (y/n): ").strip().lower() == 'y':
        Xb = add_bias(X)
        if mode == 'binary':
            p = sigmoid(Xb @ W_tr); yhat = (p >= threshold).astype(int)
            ytrue = y_or_Y
            TP = int(np.sum((yhat==1)&(ytrue==1)))
            TN = int(np.sum((yhat==0)&(ytrue==0)))
            FP = int(np.sum((yhat==1)&(ytrue==0)))
            FN = int(np.sum((yhat==0)&(ytrue==1)))
            prec = TP / (TP + FP) if TP+FP>0 else 0.0
            rec = TP / (TP + FN) if TP+FN>0 else 0.0
            f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
            acc = (TP+TN) / (TP+TN+FP+FN)
            print("Confusion: TP, FP, FN, TN = ", TP, FP, FN, TN)
            print(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
        else:
            P = softmax(Xb @ W_tr)
            yhat = np.argmax(P, axis=1)
            ytrue = np.argmax(y_or_Y, axis=1)
            from collections import Counter
            cm = np.zeros((K,K), dtype=int)
            for a,b in zip(ytrue,yhat): cm[a,b]+=1
            print("Confusion matrix (rows=true, cols=pred):\n", cm)



# ============================================================================
# PROGRAM 2: CLUSTERING
# ============================================================================

def run_clustering():
    print("\n" + "=" * 70)
    print(" CLUSTERING (K-means / Fuzzy C-means) ".center(70))
    print("=" * 70)

    def parse_matrix(rows_str: str):
        rows = [r.strip() for r in rows_str.split(',') if r.strip()]
        mat = [list(map(float, r.split())) for r in rows]
        return np.array(mat, dtype=float)

    def pairwise_sq_dists(X, C):
        x2 = np.sum(X * X, axis=1, keepdims=True)
        c2 = np.sum(C * C, axis=1, keepdims=True).T
        return x2 + c2 - 2 * (X @ C.T)

    def pairwise_dists(X, C):
        """Euclidean distances (not squared) - matches textbook formula"""
        D2 = pairwise_sq_dists(X, C)
        return np.sqrt(np.maximum(D2, 0.0))

    def kmeans_plus_plus_init(X, K):
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
        return C

    def kmeans(X, K, init_mode, manual_C=None, max_iter=200):
        if init_mode == "manual" and manual_C is not None:
            C = manual_C.astype(float).copy()
        else:
            C = kmeans_plus_plus_init(X, K)

        last_J = None
        print(f"\nInitial Centroids:\n{np.round(C, 8)}")

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

            print(f"\n--- K-means Iteration {t} ---")
            print("Centroids:\n", np.round(C_new, 8))
            print("Labels:   ", labels)
            print(f"Distortion J={J:.8f}")

            if last_J is not None and abs(last_J - J) < 1e-6:
                print("Converged: dJ < 1e-6")
                C = C_new
                break
            C, last_J = C_new, J
        return C, labels, J, t

    # =====================================================================
    # FUZZY C-MEANS - Matches course textbook exactly
    # =====================================================================

    def update_membership(data_points, centers, fuzzier=2):
        """
        Assignment Step: Fix centers, update membership
        Matches course code exactly.

        Parameters:
            data_points: ndarray of shape (n_samples, n_features)
            centers: ndarray of shape (n_clusters, n_features)
            fuzzier: fuzzifier ([1.25,2])

        Returns:
            W: ndarray of shape (n_samples, n_clusters)
        """
        n_samples = data_points.shape[0]
        n_clusters = centers.shape[0]
        W = np.zeros((n_samples, n_clusters))

        for i in range(n_samples):
            for k in range(n_clusters):
                denom = 0.0

                # Calculate ||x_i - c_k||
                dist_k = np.linalg.norm(data_points[i] - centers[k]) + 1e-10

                for j in range(n_clusters):
                    # Calculate ||x_i - c_j||
                    dist_j = np.linalg.norm(data_points[i] - centers[j]) + 1e-10
                    ratio = (dist_k / dist_j)
                    denom += ratio ** (2 / (fuzzier - 1))

                W[i, k] = 1 / denom
        return W

    def update_centers(data_points, W, fuzzier=2):
        """
        Centroid Update Step: Fix membership, update centers
        Matches course code exactly.

        Parameters:
            data_points: ndarray of shape (n_samples, n_features)
            W: ndarray of shape (n_samples, n_clusters)
            fuzzier: fuzzifier ([1.25,2])

        Returns:
            centers: ndarray of shape (n_clusters, n_features)
        """
        n_clusters = W.shape[1]
        centers = np.zeros((n_clusters, data_points.shape[1]))

        for k in range(n_clusters):
            numerator = data_points.T @ (W[:, k] ** fuzzier)
            denominator = np.sum(W[:, k] ** fuzzier)
            centers[k] = numerator / denominator
        return centers

    def fuzzy_cmeans(data_points, centers_init, fuzzier=2, max_iterations=100, tol=1e-4, verbose=True):
        """
        Fuzzy C-Means Clustering - Matches course textbook exactly.

        Parameters:
            data_points: ndarray of shape (n_samples, n_features)
            centers_init: ndarray of shape (n_clusters, n_features)
            fuzzier: fuzzifier parameter (default 2)
            max_iterations: maximum iterations (default 100)
            tol: convergence tolerance (default 1e-4)
            verbose: print iteration details

        Returns:
            centers: final cluster centers
            W: final membership matrix
            iteration: number of iterations run
        """
        centers = centers_init.copy()

        if verbose:
            print(f"\nInitial Centroids:\n{np.round(centers, 8)}")

        for iteration in range(max_iterations):
            # Assignment step: update membership matrix
            W = update_membership(data_points, centers, fuzzier)

            # Update step: compute new centers
            new_centers = update_centers(data_points, W, fuzzier)

            # Compute objective J = sum_{i,k} u_ik^m * ||x_i - c_k||^2
            n_samples = data_points.shape[0]
            n_clusters = centers.shape[0]
            J = 0.0
            for i in range(n_samples):
                for k in range(n_clusters):
                    dist_sq = np.sum((data_points[i] - new_centers[k]) ** 2)
                    J += (W[i, k] ** fuzzier) * dist_sq

            # Hard labels (argmax of membership)
            labels = np.argmax(W, axis=1)

            # Centroid change for convergence check
            centroid_change = np.linalg.norm(new_centers - centers)

            if verbose:
                print(f"\n--- FCM Iteration {iteration} ---")
                print("Centroids:\n", np.round(new_centers, 8))
                print("Membership W:\n", np.round(W, 8))
                print("Labels (hard assignment):", labels)
                print(f"Objective J = {J:.8f}")
                print(f"Centroid change = {centroid_change:.8e}")

            # Check convergence
            if centroid_change < tol:
                if verbose:
                    print(f"Converged: centroid change ({centroid_change:.2e}) < tol ({tol})")
                centers = new_centers
                break

            centers = new_centers

        return centers, W, iteration

    # ----------------------- user input & menu -----------------------
    print("Enter X rows (comma-separated):")
    X = parse_matrix(input("X = ").strip())

    print("\nChoose clustering method:")
    print("1. K-Means (manual init)")
    print("2. K-Means++ (randomized)")
    print("3. Fuzzy C-means (FCM) - Textbook version")
    method = input("Choice (Default 1): ").strip() or "1"

    if method == "3":
        # Fuzzy C-means path - textbook version
        print("Enter Initial Centroids C0 (comma-separated rows):")
        manual_C = parse_matrix(input("C0 = ").strip())
        K = manual_C.shape[0]

        m = float(input("Fuzzifier m (>1, default 2.0): ").strip() or "2.0")
        tol = float(input("Convergence tol (default 1e-4): ").strip() or "1e-4")
        max_iter = int(input("Max iterations (default 100): ").strip() or "100")

        centers, W, it = fuzzy_cmeans(X, manual_C, fuzzier=m, max_iterations=max_iter, tol=tol, verbose=True)

        print("\n" + "=" * 50)
        print(" FINAL RESULTS ".center(50))
        print("=" * 50)
        print(f"Converged after {it + 1} iterations")
        print(f"\nFinal Centroids:\n{np.round(centers, 8)}")
        print(f"\nFinal Membership Matrix W:\n{np.round(W, 8)}")
        print(f"\nFinal Labels (hard assignment): {np.argmax(W, axis=1)}")

    else:
        # K-means path
        if method == "2":
            K = int(input("K = ").strip() or "2")
            print("Using K-means++ init.")
            manual_C = None
            mode_str = "plus"
        else:
            print("Enter Initial Centroids C0 (comma-separated):")
            manual_C = parse_matrix(input("C0 = ").strip())
            K = manual_C.shape[0]
            mode_str = "manual"

        C, labels, J, it = kmeans(X, K, mode_str, manual_C)
        print(f"\nFinal distortion J={J:.8f}, converged at iter={it}")







# ============================================================================
# PROGRAM 3: GRADIENT DESCENT (Matrix Linear Regression)
# ============================================================================

def run_gradient_descent():
    print("\n" + "=" * 70)
    print(" GRADIENT DESCENT (Linear / Softmax Regression) ".center(70))
    print("=" * 70)

    def parse_matrix(s):
        rows = [r.strip() for r in s.split(',')]
        return np.array([list(map(float, r.split())) for r in rows if r], dtype=float)

    def softmax(z):
        """Numerically stable softmax"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # =====================================================================
    # LINEAR REGRESSION
    # =====================================================================
    def linear_gradient_descent(X, y, w0, lr, iters, normalize=True, print_every=1):
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1)
        w = np.asarray(w0, float).reshape(-1)
        m, n = X.shape
        factor = (2.0 / m) if normalize else 2.0

        for t in range(1, iters + 1):
            y_hat = X @ w
            grad = factor * (X.T @ (y_hat - y))
            w = w - lr * grad
            if t % print_every == 0:
                cost = np.sum((y_hat - y) ** 2)
                if normalize:
                    cost /= m
                print(f"iter {t}: w = {np.round(w, 8)}, cost = {cost:.8f}")
        return w

    # =====================================================================
    # SOFTMAX REGRESSION (Multi-class Logistic)
    # =====================================================================
    def softmax_gradient_descent(X, y, W0, lr, iters, normalize=True, print_every=1):
        """
        X: (n_samples, n_features) with bias column
        y: (n_samples,) class labels (0-indexed)
        W0: (n_features, n_classes) initial weight matrix
        """
        X = np.asarray(X, float)
        y = np.asarray(y, int).reshape(-1)
        W = np.asarray(W0, float).copy()

        n_samples = X.shape[0]
        n_classes = W.shape[1]

        # One-hot encode y
        Y_onehot = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            Y_onehot[i, y[i]] = 1

        print(f"\nY one-hot:\n{Y_onehot.astype(int)}")
        print(f"\nInitial W:\n{np.round(W, 8)}")

        for t in range(1, iters + 1):
            # Forward pass
            Z = X @ W
            P = softmax(Z)

            # Gradient
            gradient = X.T @ (P - Y_onehot)
            if normalize:
                gradient /= n_samples

            # Update
            W = W - lr * gradient

            if t % print_every == 0:
                # Cross-entropy loss
                loss = -np.sum(Y_onehot * np.log(P + 1e-10))
                if normalize:
                    loss /= n_samples
                labels = np.argmax(P, axis=1)
                accuracy = np.mean(labels == y)

                print(f"\n--- Iteration {t} ---")
                print(f"Z (logits):\n{np.round(Z, 8)}")
                print(f"P (softmax):\n{np.round(P, 8)}")
                print(f"Gradient:\n{np.round(gradient, 8)}")
                print(f"W:\n{np.round(W, 8)}")
                print(f"Loss = {loss:.8f}, Accuracy = {accuracy:.4f}")
                print(f"Predicted labels: {labels}")

        return W

    # =====================================================================
    # USER INPUT & MODE SELECTION
    # =====================================================================
    print("Select mode:")
    print("1. Linear Regression (single output)")
    print("2. Softmax Regression (multi-class classification)")
    mode = input("Choice (1/2) [Default 1]: ").strip() or "1"

    X = parse_matrix(input("X (comma separated rows, WITHOUT bias) = ").strip())

    add_bias = input("Add bias column? (y/n) [Default y]: ").strip().lower()
    if add_bias != 'n':
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        print(f"X with bias:\n{X}")

    if mode == "2":
        # -------------------- SOFTMAX REGRESSION --------------------
        print("Enter y labels (space separated):")
        y_raw = list(map(int, input("y = ").split()))

        indexing = input("Are labels 1-indexed? (y/n) [Default y]: ").strip().lower()
        if indexing != 'n':
            y = np.array(y_raw) - 1
            print(f"Converted to 0-indexed: {y}")
        else:
            y = np.array(y_raw)

        n_classes = int(input("Number of classes = "))

        print(f"Enter W0 ({X.shape[1]} rows x {n_classes} cols, comma-separated rows):")
        W0 = parse_matrix(input("W0 = ").strip())

        lr = float(input("lr = "))
        iters = int(input("iters = "))

        norm_input = input("Use 1/N scaling? (y/n) [Default y]: ").strip().lower()
        normalize = norm_input != 'n'

        W_final = softmax_gradient_descent(X, y, W0, lr, iters, normalize=normalize)

        print("\n" + "=" * 50)
        print(" FINAL RESULTS ".center(50))
        print("=" * 50)
        print(f"Final W:\n{np.round(W_final, 8)}")

        # Query specific weight
        if input("\nQuery specific w_i,j? (y/n): ").strip().lower() == 'y':
            print("Note: Uses 1-indexed columns (w_0,1 = W[0,0])")
            i = int(input("Row i = "))
            j = int(input("Col j (1-indexed) = "))
            print(f"w_{i},{j} = W[{i},{j-1}] = {W_final[i, j-1]:.8f}")

    else:
        # -------------------- LINEAR REGRESSION --------------------
        y = np.array(list(map(float, input("y (space separated) = ").split())))
        w0 = np.array(list(map(float, input("w0 (initial weights) = ").split())))
        lr = float(input("lr = "))
        iters = int(input("iters = "))

        norm_input = input("Use 1/N scaling (MSE)? (y/n) [Default y]: ").strip().lower()
        normalize = norm_input != 'n'

        w_final = linear_gradient_descent(X, y, w0, lr, iters, normalize=normalize)

        print("\n" + "=" * 50)
        print(" FINAL RESULTS ".center(50))
        print("=" * 50)
        print(f"Final w: {np.round(w_final, 8)}")

        if input("\nPredict for new X? (y/n): ").strip().lower() == "y":
            X_new = parse_matrix(input("Enter new X rows: ").strip())
            if add_bias != 'n':
                X_new = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
            y_pred = X_new @ w_final
            print(f"Predictions: {np.round(y_pred, 8)}")


# ============================================================================
# PROGRAM 4: NEURAL NETWORK
# ============================================================================
def run_neural():
    print("\n" + "=" * 70)
    print(" NEURAL NETWORK (MLP - Backprop) ".center(70))
    print("=" * 70)

    def parse_matrix(s: str):
        rows = [r.strip() for r in s.split(',') if r.strip()]
        return np.array([list(map(float, r.split())) for r in rows], dtype=float)

    def relu(z): return np.maximum(0, z)
    def relu_grad(z): return (z > 0).astype(float)
    def sigmoid(z):
        z = np.clip(z, -50, 50)
        return 1 / (1 + np.exp(-z))
    def sigmoid_grad(z):
        s = sigmoid(z); return s * (1 - s)
    def softmax(Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    ACT_FUNCS = {
        "relu": (relu, relu_grad),
        "sigmoid": (sigmoid, sigmoid_grad),
        "linear": (lambda z: z, lambda z: np.ones_like(z)),
        "softmax": (softmax, None)  # softmax gradient handled with cross-entropy
    }

    def mse(yhat, y):
        return np.mean(np.sum((yhat - y) ** 2, axis=1))

    def forward(X, weights, activations):
        caches = []
        A = X
        for i, (W, act_name) in enumerate(zip(weights, activations)):
            act, _ = ACT_FUNCS[act_name]
            A_b = np.concatenate([np.ones((A.shape[0], 1)), A], axis=1)  # add bias
            Z = A_b @ W
            if act_name == "softmax":
                A = softmax(Z)
            else:
                A = act(Z)
            caches.append((A_b, Z, act_name))
        return A, caches

    def backward(X, Y, Yhat, weights, caches, activations, loss_type="mse"):
        grads = []
        N = X.shape[0]
        if loss_type == "mse":
            dA = (2.0 / N) * (Yhat - Y)
        else:
            # cross-entropy + softmax final layer => simplified gradient at last layer:
            # dZ_last = (Yhat - Y) / N
            dA = (Yhat - Y) / N

        for i in reversed(range(len(weights))):
            A_b, Z, act_name = caches[i]
            if act_name == "softmax" and loss_type != "mse":
                # dZ is dA already (we used the simplification above)
                dZ = dA
            else:
                _, act_grad = ACT_FUNCS[act_name]
                dZ = dA * act_grad(Z)
            gW = A_b.T @ dZ
            grads.insert(0, gW)
            if i > 0:
                dA_b = dZ @ weights[i].T
                dA = dA_b[:, 1:]
        return grads

    def init_weights_for_layer(in_dim, out_dim, init_mode, seed=None, act_name=None):
        # in_dim = previous layer size (no bias)
        rows = in_dim + 1  # include bias row
        cols = out_dim
        if seed is not None and seed != "":
            np.random.seed(int(seed))
        if init_mode == "zeros":
            return np.zeros((rows, cols))
        if init_mode == "xavier":
            # Xavier normal
            std = np.sqrt(2.0 / (in_dim + out_dim))
            return np.random.randn(rows, cols) * std
        if init_mode == "he":
            std = np.sqrt(2.0 / max(1, in_dim))
            return np.random.randn(rows, cols) * std
        if init_mode == "random":
            return np.random.randn(rows, cols)
        # fallback (should be handled earlier)
        return np.zeros((rows, cols))

    def train_step(X, Y, weights, activations, lr=0.1, loss_type="mse"):
        Yhat, caches = forward(X, weights, activations)
        loss = mse(Yhat, Y) if loss_type == "mse" else -np.mean(np.sum(Y * np.log(np.clip(Yhat, 1e-12, 1 - 1e-12)), axis=1))
        grads = backward(X, Y, Yhat, weights, caches, activations, loss_type=loss_type)
        new_weights = [W - lr * g for W, g in zip(weights, grads)]
        return new_weights, loss, Yhat

    # ---------------------------
    # User inputs
    # ---------------------------
    print("Enter X rows separated by commas NO BIAS (e.g. '2 1, 5 1'):")
    X = parse_matrix(input("X = ").strip())
    print("Enter Y rows (targets), same #rows as X:")
    Y = parse_matrix(input("Y = ").strip())

    L = int(input("\nNumber of layers (1-5) [default 2]: ").strip() or "2")
    activations = []
    weights = []

    prev_dim = X.shape[1]
    # For each layer, ask activation and how to init W
    for i in range(L):
        print(f"\nLayer {i + 1} (input dim = {prev_dim}):")
        act = input("  Activation ('relu','sigmoid','linear','softmax') [default relu]: ").strip().lower() or "relu"
        out_dim = int(input("  Output dimension (#neurons) [required]: ").strip())
        activations.append(act)

        # initial weight choice for this layer
        print("  Weight init options:")
        print("   1) zeros")
        print("   2) xavier")
        print("   3) he")
        print("   4) random (normal)")
        print("   5) manual (paste matrix rows comma-separated; rows must equal bias+input_dim)")
        init_choice = input("  Choose init (1-5) [default 2]: ").strip() or "2"

        W = None
        if init_choice == "1":
            W = init_weights_for_layer(prev_dim, out_dim, "zeros")
        elif init_choice == "2":
            seed = input("  Seed for xavier init (blank = random): ").strip() or ""
            W = init_weights_for_layer(prev_dim, out_dim, "xavier", seed=seed)
        elif init_choice == "3":
            seed = input("  Seed for he init (blank = random): ").strip() or ""
            W = init_weights_for_layer(prev_dim, out_dim, "he", seed=seed)
        elif init_choice == "4":
            seed = input("  Seed for random init (blank = random): ").strip() or ""
            if seed: np.random.seed(int(seed))
            W = init_weights_for_layer(prev_dim, out_dim, "random")
        elif init_choice == "5":
            print(f"  Enter initial W^{i+1} as comma-separated rows (rows must be {prev_dim+1}):")
            raw = input("  W = ").strip()
            try:
                W_try = parse_matrix(raw)
                if W_try.shape[0] != prev_dim + 1:
                    print(f"  Warning: pasted W has {W_try.shape[0]} rows but expected {prev_dim+1}.")
                    # allow it but will try to reshape if possible
                if W_try.shape[1] != out_dim:
                    print(f"  Warning: pasted W has {W_try.shape[1]} cols but expected {out_dim}.")
                W = W_try
            except Exception as e:
                print("  Error parsing matrix:", e)
                print("  Falling back to zeros init for this layer.")
                W = init_weights_for_layer(prev_dim, out_dim, "zeros")
        else:
            # default xavier
            seed = input("  Seed for xavier init (blank = random): ").strip() or ""
            W = init_weights_for_layer(prev_dim, out_dim, "xavier", seed=seed)

        # final sanity: ensure W shape matches (prev_dim+1, out_dim)
        if W is None:
            W = init_weights_for_layer(prev_dim, out_dim, "xavier")
        if W.shape != (prev_dim + 1, out_dim):
            # try to reshape if total elements match
            try:
                flat = W.flatten()
                expected = (prev_dim + 1) * out_dim
                if flat.size == expected:
                    W = flat.reshape(prev_dim + 1, out_dim)
                    print(f"  Reshaped pasted W -> {(prev_dim + 1, out_dim)}")
                else:
                    # fallback: create proper-shaped random / zeros depending on sign of W
                    print(f"  Can't reshape pasted W to {(prev_dim + 1, out_dim)}. Using xavier instead.")
                    W = init_weights_for_layer(prev_dim, out_dim, "xavier")
            except Exception:
                W = init_weights_for_layer(prev_dim, out_dim, "xavier")

        prev_dim = W.shape[1]
        weights.append(W)

    # training hyperparams
    lr = float(input("\nLearning rate (default 0.1): ").strip() or "0.1")
    iters = int(input("Iterations / epochs (e.g. 1) [default 1]: ").strip() or "1")
    loss_type = input("Loss ('mse' or 'ce')[regression=mse,classification=ce]: ").strip().lower() or "mse"

    print("\nStarting training...")
    for t in range(1, iters + 1):
        weights, loss, Yhat = train_step(X, Y, weights, activations, lr=lr, loss_type=loss_type)
        print(f"\nIter {t}: loss = {loss:.8f}")
        print("Predicted Yhat (Forward Pass):\n", np.round(Yhat, 8))

    for i, W in enumerate(weights, 1):
        print(f"\nUpdated W^{i} (shape {W.shape}):\n", np.round(W, 8))

    if input("\nPredict for new X values? (y/n): ").strip().lower() == "y":
        X_new = parse_matrix(input("Enter new X rows (NO BIAS, comma-separated): ").strip())
        Yhat_new, _ = forward(X_new, weights, activations)
        print("\nPredictions on new data:")
        print("Y_pred =\n", np.round(Yhat_new, 8))




# ============================================================================
# PROGRAM 5: REGRESSION (OLS, Ridge, Polynomial) - WITH PEARSON R
# ============================================================================

def run_regression():
    print("\n" + "=" * 70)
    print(" REGRESSION (OLS, Ridge, Polynomial) ".center(70))
    print("=" * 70)

    def parse_matrix(rows_str):
        rows = [r.strip() for r in rows_str.split(',') if r.strip()]
        mat = [list(map(float, r.split())) for r in rows]
        return np.array(mat, dtype=float)

    def _as_2d(a):
        a = np.asarray(a, dtype=float)
        if a.ndim == 0:
            a = a.reshape(1, 1)
        elif a.ndim == 1:
            a = a.reshape(-1, 1)
        return a

    def _pearson_r(Y, Y_pred):
        """Calculate Pearson correlation coefficient"""
        Y = Y.flatten()
        Y_pred = Y_pred.flatten()

        mean_y = np.mean(Y)
        mean_pred = np.mean(Y_pred)

        numerator = np.sum((Y - mean_y) * (Y_pred - mean_pred))
        denominator = np.sqrt(np.sum((Y - mean_y) ** 2) *
                              np.sum((Y_pred - mean_pred) ** 2))

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
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
        r = _pearson_r(Y, Y_pred)
        return mse, rmse, mae, r2, adj_r2, r

    def _print_metrics(tag, mse, rmse, mae, r2, adj_r2, r):
        print(f"\n--- {tag} | Metrics ---")
        print(f"MSE:         {mse:.8f}")
        print(f"RMSE:        {rmse:.8f}")
        print(f"MAE:         {mae:.8f}")
        print(f"R2:          {r2:.8f}")
        print("Adjusted R2:  " + (f"{adj_r2:.8f}" if not np.isnan(adj_r2) else "N/A"))
        print(f"Pearson r:   {r:.8f}")

    def _print_model(tag, W, b):
        print(f"\n=== {tag} ===")
        print(f"Model: Y = XW + b")
        print("W (coeffs) =\n", np.round(W, 8))
        print("b (intercept) =", np.round(b, 8))

    def ols_add_intercept(X, Y):
        X = _as_2d(X);
        Y = _as_2d(Y)
        n, p = X.shape
        Xa = np.hstack([X, np.ones((n, 1))])
        Theta, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
        W, b = Theta[:-1, :], Theta[-1, :]
        Yp = Xa @ Theta
        mets = _metrics(Y, Yp, p=p)
        return W, b, Yp, mets

    def ridge_add_intercept(X, Y, alpha, penalize_bias=False):
        X = _as_2d(X);
        Y = _as_2d(Y)
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
        return W, b, Yp, mets

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
            if self.combos is None: self.fit(X)
            n = X.shape[0]
            out = []
            for combo in self.combos:
                col = np.ones(n)
                for idx in combo: col *= X[:, idx]
                out.append(col.reshape(-1, 1))
            return np.hstack(out) if out else np.empty((n, 0))

    print("Enter X rows separated by commas:")
    X = parse_matrix(input("X = ").strip())
    print("Enter Y rows:")
    Y = parse_matrix(input("Y = ").strip())

    print("\nChoose model: 1=OLS  2=Ridge  3=Polynomial")
    choice = input("Enter 1 / 2 / 3: ").strip()
    poly = None
    Wa, ba = None, None

    if choice == "1":
        Wa, ba, Ypa, ma = ols_add_intercept(X, Y)
        _print_model("OLS", Wa, ba)
        _print_metrics("OLS", *ma)
    elif choice == "2":
        alpha = float(input("Enter ridge alpha: "))
        pen_bias = input("Penalize Intercept (Use for simple formulas)? (y/n) [Default n]: ").strip().lower() == 'y'

        Wa, ba, Ypa, ma = ridge_add_intercept(X, Y, alpha, penalize_bias=pen_bias)
        _print_model(f"Ridge (alpha={alpha}, pen_bias={pen_bias})", Wa, ba)
        _print_metrics("Ridge", *ma)

    elif choice == "3":
        degree = int(input("Enter polynomial degree: "))
        use_ridge = input("Add ridge? (y/n): ").strip().lower() == "y"
        alpha = float(input("Enter ridge alpha: ")) if use_ridge else 0.0

        poly = PolynomialFeatures(degree).fit(X)
        Phi = poly.transform(X)

        if use_ridge and alpha > 0:
            pen_bias = input("Penalize Intercept? (y/n): ").strip().lower() == 'y'
            Wa, ba, Ypa, ma = ridge_add_intercept(Phi, Y, alpha, penalize_bias=pen_bias)
        else:
            Wa, ba, Ypa, ma = ols_add_intercept(Phi, Y)

        # Print with labeled polynomial terms
        from collections import Counter
        print(f"\n=== Poly ===")
        print(f"Model: Y = XW + b")
        print("W (coeffs):")
        w_rounded = np.round(Wa, 8)
        labels = []
        for combo in poly.combos:
            counts = Counter(combo)
            parts = []
            for idx in sorted(counts):
                power = counts[idx]
                name = f"x{idx + 1}"
                if power > 1:
                    name += f"^{power}"
                parts.append(name)
            labels.append("*".join(parts))
        max_label_len = max(len(l) for l in labels)
        for i, label in enumerate(labels):
            print(f"  {label:<{max_label_len}} : {w_rounded[i].tolist()}")
        print("b (intercept) =", np.round(ba, 8))
        _print_metrics("Polynomial", *ma)

    if input("\nPredict for new x values? (y/n): ").strip().lower() == "y":
        Xn = parse_matrix(input("Enter new X rows: ").strip())
        if choice == "3" and poly:
            Pa = poly.transform(Xn) @ Wa + ba
        else:
            Pa = Xn @ Wa + ba
        print("\nPredictions:", np.round(Pa, 8))


# ============================================================================
# PROGRAM 6: DECISION TREE & RANDOM FOREST (CLASSIFICATION + REGRESSION)
# ============================================================================

def run_tree_forest():
    print("\n" + "=" * 70)
    print(" DECISION TREE & RANDOM FOREST ".center(70))
    print("=" * 70)

    def parse_matrix(s):
        rows = [r.strip() for r in s.split(',') if r.strip()]
        return np.array([list(map(float, r.split())) for r in rows])

    def parse_labels(s):
        y = np.array(list(map(float, s.split())))
        return y

    # ========== CLASSIFICATION TREE FUNCTIONS ==========
    def gini(y):
        _, c = np.unique(y, return_counts=True)
        p = c / c.sum()
        return 1 - np.sum(p * p)

    def entropy(y):
        _, c = np.unique(y, return_counts=True)
        p = c / c.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    def impurity_classification(y, crit):
        return gini(y) if crit == "gini" else entropy(y)

    # ========== REGRESSION TREE FUNCTIONS ==========
    def mse_impurity(y):
        """MSE impurity for regression: Var(y) = mean((y - y_mean)^2)"""
        if len(y) == 0:
            return 0.0
        return np.var(y)

    def best_split_classification(X, y, crit, feat_idx, thr_map=None):
        """Find best split for classification.
        thr_map: dict mapping feature index -> list/array of thresholds to try.
        If thr_map is None or a feature is not present, midpoints between unique values are used.
        """
        parent = impurity_classification(y, crit)
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
                impL = impurity_classification(y[L], crit)
                impR = impurity_classification(y[R], crit)
                child = (L.sum() * impL + R.sum() * impR) / n
                gain = parent - child
                if gain > best_gain:
                    best_gain = gain
                    best = j
                    best_thr = thr
                    best_child = child
        return best, best_thr, best_gain, parent, best_child

    def best_split_regression(X, y, feat_idx, thr_map=None):
        """Find best split for regression (minimize MSE)."""
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
                gain = parent_mse - child_mse

                if gain > best_gain:
                    best_gain = gain
                    best = j
                    best_thr = thr
                    best_child_mse = child_mse
        return best, best_thr, best_gain, parent_mse, best_child_mse

    class Node:
        def __init__(self, f=None, t=None, l=None, r=None, pred=None):
            self.f = f
            self.t = t
            self.l = l
            self.r = r
            self.pred = pred

        def leaf(self):
            return self.pred is not None

    def majority(y):
        """For classification: return most common class"""
        v, c = np.unique(y, return_counts=True)
        return v[np.argmax(c)]

    def mean_value(y):
        """For regression: return mean"""
        return np.mean(y)

    def build_classification_tree(X, y, d, maxd, mins, crit, feat_sub, thr_map=None):
        """Build classification tree"""
        if d >= maxd or len(np.unique(y)) == 1 or len(y) < mins:
            return Node(pred=majority(y))

        nfeat = X.shape[1]
        if feat_sub < nfeat:
            idx = np.random.choice(nfeat, feat_sub, replace=False)
        else:
            idx = np.arange(nfeat)

        f, thr, g, parent, ch = best_split_classification(X, y, crit, idx, thr_map)

        if f is None:
            return Node(pred=majority(y))

        L = X[:, f] <= thr
        return Node(f, thr,
                    build_classification_tree(X[L], y[L], d + 1, maxd, mins, crit, feat_sub, thr_map),
                    build_classification_tree(X[~L], y[~L], d + 1, maxd, mins, crit, feat_sub, thr_map))

    def build_regression_tree(X, y, d, maxd, mins, feat_sub, thr_map=None):
        """Build regression tree using MSE"""
        if d >= maxd or len(y) < mins:
            return Node(pred=mean_value(y))

        # Check if all y values are the same (no variance)
        if np.var(y) < 1e-10:
            return Node(pred=mean_value(y))

        nfeat = X.shape[1]
        if feat_sub < nfeat:
            idx = np.random.choice(nfeat, feat_sub, replace=False)
        else:
            idx = np.arange(nfeat)

        f, thr, g, parent_mse, child_mse = best_split_regression(X, y, idx, thr_map)

        if f is None:
            return Node(pred=mean_value(y))

        L = X[:, f] <= thr
        return Node(f, thr,
                    build_regression_tree(X[L], y[L], d + 1, maxd, mins, feat_sub, thr_map),
                    build_regression_tree(X[~L], y[~L], d + 1, maxd, mins, feat_sub, thr_map))

    def predict_tree(root, X):
        """Predict for both classification and regression"""
        out = []
        for x in X:
            n = root
            while not n.leaf():
                n = n.l if x[n.f] <= n.t else n.r
            out.append(n.pred)
        return np.array(out)

    class Forest:
        def __init__(self, T, depth, mins, task, crit_or_none, feat_mode, thr_map=None):
            self.T = T
            self.depth = depth
            self.mins = mins
            self.task = task  # 'classification' or 'regression'
            self.crit = crit_or_none  # Only for classification
            self.fm = feat_mode
            self.trees = []
            self.thr_map = thr_map

        def _sub(self, k):
            if self.fm == "sqrt": return max(1, int(np.sqrt(k)))
            if self.fm == "log2": return max(1, int(np.log2(k)))
            if isinstance(self.fm, int): return min(k, self.fm)
            return k

        def fit(self, X, y):
            n = X.shape[0]
            k = X.shape[1]
            sub = self._sub(k)
            self.trees = []
            for _ in range(self.T):
                idx = np.random.randint(0, n, n)
                Xb, yb = X[idx], y[idx]
                if self.task == 'classification':
                    tree = build_classification_tree(Xb, yb, 0, self.depth, self.mins, self.crit, sub, self.thr_map)
                else:  # regression
                    tree = build_regression_tree(Xb, yb, 0, self.depth, self.mins, sub, self.thr_map)
                self.trees.append(tree)

        def predict(self, X):
            allp = np.array([predict_tree(t, X) for t in self.trees])
            if self.task == 'classification':
                # Majority vote
                out = []
                for j in range(X.shape[0]):
                    v, c = np.unique(allp[:, j], return_counts=True)
                    out.append(v[np.argmax(c)])
                return np.array(out)
            else:  # regression
                # Average predictions
                return np.mean(allp, axis=0)

    # ========== Threshold parsing utility ==========
    def parse_thresholds(s, n_features):
        """Parse user thresholds string into a dict: {feat_idx: [thr1, thr2, ...]}
        Expected formats:
          """
        if not s or s.strip() == "":
            return None
        s = s.strip()
        thr_map = {}
        # allow separators ; or , between entries
        parts = [p.strip() for p in re.split('[;,]', s) if p.strip()]
        for p in parts:
            if ':' in p:
                fi, vals = p.split(':', 1)
                fi = int(fi.strip())
                nums = [float(x) for x in vals.split() if x.strip()]
                thr_map[fi] = nums
            else:
                # no feature index: apply to all features
                nums = [float(x) for x in p.split() if x.strip()]
                for fi in range(n_features):
                    thr_map.setdefault(fi, nums)
        # sanitize: keep only valid feature indices
        thr_map = {k: v for k, v in thr_map.items() if 0 <= k < n_features and len(v) > 0}
        return thr_map if thr_map else None

    X = parse_matrix(input("X = ").strip())
    y = parse_labels(input("y (space separated) = ").strip())

    # Determine task type
    print("\nTask type:")
    print("1 = Classification (discrete labels)")
    print("2 = Regression (continuous values)")
    task_choice = input("Task = ").strip()

    if task_choice == "1":
        task = "classification"
        crit = input("Criterion (gini/entropy): ").strip() or "gini"
        y = y.astype(int)  # Ensure integer labels for classification
    else:
        task = "regression"
        crit = None  # Not used for regression

    # Ask user whether to auto-find thresholds or input manually
    thr_map = None
    # Always allow manual thresholds, even for single-feature data.
    print("Threshold options:")
    print("1 = Auto-find thresholds (default)")
    print("2 = Enter thresholds manually")
    thr_choice = input("Threshold choice = ").strip() or "1"
    if thr_choice == "2":
        print("Enter thresholds per feature. Examples:")
        print("  0: 1.2 3.4, 1: 0.5 2.1    -> thresholds for feature 0 and 1")
        print("  0.8 1.6               -> same thresholds applied to all features")
        print("If your data has only one feature you may simply enter: 0.5 1.2  (or just numbers)")
        thr_s = input("Thresholds = ")
        thr_map = parse_thresholds(thr_s, X.shape[1])

    print("\n1 = best split only (root analysis)")
    print("2 = train decision tree")
    print("3 = train random forest")
    mode = input("Mode = ").strip()

    if mode == "1":
        # Root split analysis
        if task == "classification":
            f, thr, g, p, c = best_split_classification(X, y, crit, np.arange(X.shape[1]), thr_map)
            print(f"\nParent Impurity ({crit}): {p:.8f}")
            if f is not None:
                print(f"Best Split: Feature index {f}, Threshold {thr:.8f}")
                print(f"Gain: {g:.8f}")
                print(f"Child Impurity: {c:.8f}")
            else:
                print("No split possible.")
        else:  # regression
            f, thr, g, p_mse, c_mse = best_split_regression(X, y, np.arange(X.shape[1]), thr_map)
            print(f"\nParent MSE: {p_mse:.8f}")
            if f is not None:
                print(f"Best Split: Feature index {f}, Threshold {thr:.8f}")
                print(f"MSE Reduction: {g:.8f}")
                print(f"Child MSE: {c_mse:.8f}")
            else:
                print("No split possible.")

    elif mode == "2":
        # Single decision tree
        depth = int(input("Max depth: ").strip() or "3")
        mins = int(input("Min samples split: ").strip() or "2")
        sub = X.shape[1]

        if task == "classification":
            tree = build_classification_tree(X, y, 0, depth, mins, crit, sub, thr_map)
        else:
            tree = build_regression_tree(X, y, 0, depth, mins, sub, thr_map)

        print("Tree built.")

        # Show training predictions
        train_pred = predict_tree(tree, X)
        if task == "classification":
            acc = np.mean(train_pred == y)
            print(f"Training Accuracy: {acc:.4f}")
        else:
            mse = np.mean((train_pred - y) ** 2)
            rmse = np.sqrt(mse)
            print(f"Training MSE: {mse:.8f}")
            print(f"Training RMSE: {rmse:.8f}")

        if input("\nPredict new X? (y/n): ").lower() == "y":
            Xn = parse_matrix(input("New X = ").strip())
            pred = predict_tree(tree, Xn)
            print("Predictions:", np.round(pred, 8))

    else:
        # Random forest
        T = int(input("Number of Trees: ").strip() or "10")
        depth = int(input("Max Depth: ").strip() or "3")
        fm = input("Max features (sqrt/log2): ").strip() or "sqrt"

        rf = Forest(T, depth, 2, task, crit, fm, thr_map)
        rf.fit(X, y)
        print("Forest built.")

        # Show training predictions
        train_pred = rf.predict(X)
        if task == "classification":
            acc = np.mean(train_pred == y)
            print(f"Training Accuracy: {acc:.4f}")
        else:
            mse = np.mean((train_pred - y) ** 2)
            rmse = np.sqrt(mse)
            print(f"Training MSE: {mse:.8f}")
            print(f"Training RMSE: {rmse:.8f}")

        if input("\nPredict new X? (y/n): ").lower() == "y":
            Xn = parse_matrix(input("New X = ").strip())
            pred = rf.predict(Xn)
            print("Predictions:", np.round(pred, 8))



# ============================================================================
# PROGRAM 7: GENERIC GRADIENT DESCENT (Calculus)
# ============================================================================

def run_generic_gradient_descent():
    print("" + "=" * 70)
    print(" GENERIC CALCULUS GRADIENT DESCENT ".center(70))
    print("=" * 70)

    print("1 = Minimize C(w)")
    print("2 = Minimize C(x,y)")
    print("3 = Minimize C(a,b,c)")
    print("4 = Minimize C(any number of variables)")
    mode = input("Mode: ").strip()
    h = 1e-6

    # =============================
    # Mode 1: Single Variable w
    # =============================
    if mode == "1":
        expr = input("Enter C(w) (e.g., 'w**2 + 2*w'): ").strip()
        w = float(input("initial w = "))
        lr = float(input("lr = "))
        it = int(input("iters = "))
        for t in range(1, it + 1):
            C = eval(expr, {"w": w, "math": math})
            Cp = eval(expr, {"w": w + h, "math": math})
            Cm = eval(expr, {"w": w - h, "math": math})
            g = (Cp - Cm) / (2 * h)
            w -= lr * g
            print(f"iter {t}: w={w:.8f}, C={C:.8f}, grad={g:.8f}")
        print("Final w:", w)
        return

    # =============================
    # Mode 2: Two Variables x, y
    # =============================
    if mode == "2":
        expr = input("Enter C(x,y) (e.g., 'x**2 + y**2'): ").strip()
        x = float(input("x0 = "))
        y = float(input("y0 = "))
        lr = float(input("lr = "))
        it = int(input("iters = "))
        for t in range(1, it + 1):
            C = eval(expr, {"x": x, "y": y, "math": math})

            Cxp = eval(expr, {"x": x + h, "y": y, "math": math})
            Cxm = eval(expr, {"x": x - h, "y": y, "math": math})
            g_x = (Cxp - Cxm) / (2 * h)

            Cyp = eval(expr, {"x": x, "y": y + h, "math": math})
            Cym = eval(expr, {"x": x, "y": y - h, "math": math})
            g_y = (Cyp - Cym) / (2 * h)

            x -= lr * g_x
            y -= lr * g_y
            print(f"iter {t}: x={x:.8f}, y={y:.8f}, C={C:.8f}, grad=({g_x:.8f},{g_y:.8f})")
        print(f"Final (x,y): {x}, {y}")
        return

    # =============================
    # Mode 3: Three Variables a, b, c
    # =============================
    if mode == "3":
        expr = input("Enter C(a,b,c) (e.g., 'a**2 + b**2 + c**2'): ").strip()
        a = float(input("a0 = "))
        b = float(input("b0 = "))
        c = float(input("c0 = "))
        lr = float(input("lr = "))
        it = int(input("iters = "))

        for t in range(1, it + 1):
            C = eval(expr, {"a": a, "b": b, "c": c, "math": math})

            # gradient wrt a
            Cap = eval(expr, {"a": a + h, "b": b, "c": c, "math": math})
            Cam = eval(expr, {"a": a - h, "b": b, "c": c, "math": math})
            g_a = (Cap - Cam) / (2 * h)

            # gradient wrt b
            Cbp = eval(expr, {"a": a, "b": b + h, "c": c, "math": math})
            Cbm = eval(expr, {"a": a, "b": b - h, "c": c, "math": math})
            g_b = (Cbp - Cbm) / (2 * h)

            # gradient wrt c
            Ccp = eval(expr, {"a": a, "b": b, "c": c + h, "math": math})
            Ccm = eval(expr, {"a": a, "b": b, "c": c - h, "math": math})
            g_c = (Ccp - Ccm) / (2 * h)

            # update
            a -= lr * g_a
            b -= lr * g_b
            c -= lr * g_c

            print(
                f"iter {t}: a={a:.8f}, b={b:.8f}, c={c:.8f}, C={C:.8f}, "
                f"grad=({g_a:.8f},{g_b:.8f},{g_c:.8f})"
            )

        print(f"Final (a,b,c): {a}, {b}, {c}")
        return

    # =============================
    # Mode 4: Arbitrary N variables
    # =============================
    if mode == "4":
        expr = input("Enter cost C( ... ) using your variable names (e.g., 'x1**2 + math.sin(x2)'): ").strip()
        var_names_s = input("Enter variable names (space-separated), e.g. 'x1 x2 x3': ").strip()
        var_names = [v.strip() for v in var_names_s.split() if v.strip()]
        if len(var_names) == 0:
            print("No variables given. Exiting.")
            return

        inits_s = input(f"Enter initial values for {len(var_names)} variables (space-separated): ").strip()
        inits = [float(x) for x in inits_s.split()]
        if len(inits) != len(var_names):
            print("Number of initials does not match number of variables. Exiting.")
            return

        # make a dict of current values
        vals = {name: float(val) for name, val in zip(var_names, inits)}
        lr = float(input("lr = "))
        it = int(input("iters = "))

        for t in range(1, it + 1):
            # evaluate cost at current point
            C = eval(expr, {**vals, "math": math})

            grads = {}
            # central difference for each variable
            for name in var_names:
                orig = vals[name]
                # plus
                vals[name] = orig + h
                Cp = eval(expr, {**vals, "math": math})
                # minus
                vals[name] = orig - h
                Cm = eval(expr, {**vals, "math": math})
                # restore
                vals[name] = orig
                grads[name] = (Cp - Cm) / (2 * h)

            # update all variables
            for name in var_names:
                vals[name] -= lr * grads[name]

            # pretty print
            vals_print = ", ".join(f"{name}={vals[name]:.8f}" for name in var_names)
            grad_print = ", ".join(f"{grads[name]:.8f}" for name in var_names)
            print(f"iter {t}: {vals_print}, C={C:.8f}, grad=({grad_print})")

        final_print = ", ".join(f"{name}={vals[name]}" for name in var_names)
        print(f"Final: {final_print}")
        return

    print("Invalid mode.")



# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    while True:
        print("\n" + "=" * 70)
        print(" MAIN MENU ".center(70))
        print("=" * 70)
        print("1. Classification (Binary / Multiclass)")
        print("2. Clustering (K-Means / Fuzzy C-Means)")
        print("3. Linear & Softmax GD (Train W with GD iteratively)")
        print("4. Neural Network (MLP Forward & Backprop)")
        print("5. Regression (OLS / Ridge / Polynomial)")
        print("6. Decision Tree & Random Forest (Classification + Regression)")
        print("7. Cost Function Minimizer (Custom Formula)")
        print("0. Exit")

        ch = input("Choice = ").strip()
        if ch == "0":
            print("Good luck with the exam!")
            break
        elif ch == "1":
            run_classification()
        elif ch == "2":
            run_clustering()
        elif ch == "3":
            run_gradient_descent()
        elif ch == "4":
            run_neural()
        elif ch == "5":
            run_regression()
        elif ch == "6":
            run_tree_forest()
        elif ch == "7":
            run_generic_gradient_descent()

        input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting. Good luck!")