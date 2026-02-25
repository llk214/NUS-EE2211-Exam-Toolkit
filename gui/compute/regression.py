import numpy as np
import io
from itertools import combinations_with_replacement

from gui.utils import parse_matrix


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


def _poly_term_labels(combos):
    """Convert combo tuples to human-readable polynomial term labels.
    e.g. (0,) -> 'x1', (0,0) -> 'x1^2', (0,1) -> 'x1*x2'
    """
    from collections import Counter
    labels = []
    for combo in combos:
        counts = Counter(combo)
        parts = []
        for idx in sorted(counts):
            power = counts[idx]
            name = f"x{idx + 1}"
            if power > 1:
                name += f"^{power}"
            parts.append(name)
        labels.append("*".join(parts))
    return labels


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
        poly_labels = _poly_term_labels(poly.combos)
        out.write(f"=== Polynomial (degree={degree}, alpha={alpha}) ===\n")

    out.write(f"Model: Y = XW + b\n")
    if model == 'polynomial':
        out.write("W (coeffs):\n")
        w_rounded = np.round(W, 8)
        max_label_len = max(len(l) for l in poly_labels)
        for i, label in enumerate(poly_labels):
            out.write(f"  {label:<{max_label_len}} : {w_rounded[i].tolist()}\n")
        result['poly_labels'] = poly_labels
    else:
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
