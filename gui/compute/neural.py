import numpy as np
import io

from gui.utils import parse_matrix
from gui.compute.classification import sigmoid, softmax


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
