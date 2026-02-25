import numpy as np
import io

from gui.utils import parse_matrix
from gui.compute.classification import softmax


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
