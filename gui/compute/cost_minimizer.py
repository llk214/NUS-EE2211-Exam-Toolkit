import numpy as np
import math
import io

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
