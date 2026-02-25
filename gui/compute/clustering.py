import numpy as np
import io

from gui.utils import parse_matrix


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
