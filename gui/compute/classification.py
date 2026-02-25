import numpy as np
import io

from gui.utils import parse_matrix


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
