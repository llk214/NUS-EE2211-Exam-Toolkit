"""
Linear Programming compute module.
Uses scipy.optimize.linprog (same as optimization.py).
Supports maximize/minimize, <=/>=/= constraints, non-negativity, and integer LP.
"""

import numpy as np

try:
    from scipy.optimize import linprog as _linprog
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class LinearProgramSolver:
    """LP solver using scipy.optimize.linprog — reused from optimization.py."""

    def __init__(self, num_variables=2):
        if num_variables < 1 or num_variables > 7:
            raise ValueError("Number of variables must be between 1 and 7")
        self.num_variables = num_variables
        self.constraints = []
        self.objective = None
        self.is_maximize = True
        self.bounds = [(0, None)] * num_variables
        self.variable_names = [f'x{i + 1}' for i in range(num_variables)]

    def parse_constraint(self, constraint_str):
        constraint_str = constraint_str.strip()
        if not constraint_str:
            return None
        if '<=' in constraint_str:
            parts = constraint_str.split('<=')
            ineq = '<='
        elif '>=' in constraint_str:
            parts = constraint_str.split('>=')
            ineq = '>='
        elif '=' in constraint_str and '!=' not in constraint_str:
            parts = constraint_str.split('=')
            ineq = '='
        elif '<' in constraint_str:
            parts = constraint_str.split('<')
            ineq = '<'
        elif '>' in constraint_str:
            parts = constraint_str.split('>')
            ineq = '>'
        else:
            raise ValueError(f"No valid inequality found in: {constraint_str}")
        coeffs = [float(x) for x in parts[0].strip().split()]
        if len(coeffs) != self.num_variables:
            raise ValueError(f"Expected {self.num_variables} coefficients, got {len(coeffs)}")
        bound = float(parts[1].strip())
        return coeffs, ineq, bound

    def set_objective(self, coeffs, maximize=True):
        self.is_maximize = maximize
        self.objective = coeffs

    def add_constraint(self, coeffs, ineq_type, bound):
        self.constraints.append((coeffs, bound, ineq_type))

    def solve(self, integer=False):
        if not HAS_SCIPY:
            return {'status': 'error', 'type': 'Error',
                    'optimal_solution': None, 'optimal_value': None,
                    'message': 'scipy is required. Install with: pip install scipy'}
        if not self.objective or not self.constraints:
            return {'status': 'error', 'type': 'Error',
                    'optimal_solution': None, 'optimal_value': None,
                    'message': 'No objective or constraints defined'}

        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        for coeffs, bound, ineq_type in self.constraints:
            if ineq_type in ('<=', '<'):
                A_ub.append(coeffs)
                b_ub.append(bound)
            elif ineq_type in ('>=', '>'):
                A_ub.append([-c for c in coeffs])
                b_ub.append(-bound)
            elif ineq_type in ('=', '=='):
                A_eq.append(coeffs)
                b_eq.append(bound)

        A_ub_arr = np.array(A_ub) if A_ub else None
        b_ub_arr = np.array(b_ub) if b_ub else None
        A_eq_arr = np.array(A_eq) if A_eq else None
        b_eq_arr = np.array(b_eq) if b_eq else None

        c = np.array(self.objective)
        if self.is_maximize:
            c = -c

        if integer:
            return self._solve_integer(c, A_ub_arr, b_ub_arr, A_eq_arr, b_eq_arr)

        result = _linprog(c, A_ub=A_ub_arr, b_ub=b_ub_arr,
                          A_eq=A_eq_arr, b_eq=b_eq_arr,
                          bounds=self.bounds, method='highs')
        return self._classify_result(result)

    def _solve_integer(self, c, A_ub, b_ub, A_eq, b_eq):
        """Integer LP via branch-and-bound."""
        best = [None, None]  # [x, obj_value]

        def _solve_node(extra_ub_A, extra_ub_b):
            if extra_ub_A:
                a = np.vstack([A_ub, extra_ub_A]) if A_ub is not None else np.array(extra_ub_A)
                b = np.concatenate([b_ub, extra_ub_b]) if b_ub is not None else np.array(extra_ub_b)
            else:
                a, b = A_ub, b_ub
            res = _linprog(c, A_ub=a, b_ub=b, A_eq=A_eq, b_eq=b_eq,
                           bounds=self.bounds, method='highs')
            return res

        n = self.num_variables
        stack = [([], [])]
        for _ in range(5000):
            if not stack:
                break
            ex_A, ex_b = stack.pop()
            res = _solve_node(ex_A, ex_b)
            if res.status != 0:
                continue
            obj = -res.fun if self.is_maximize else res.fun
            if best[1] is not None:
                if (self.is_maximize and obj <= best[1]) or (not self.is_maximize and obj >= best[1]):
                    continue
            x = res.x
            if all(abs(x[i] - round(x[i])) < 1e-6 for i in range(n)):
                best[0] = x[:n].copy()
                best[1] = obj
                continue
            # Branch on most fractional
            fracs = [(abs(x[i] - round(x[i])), i) for i in range(n)
                     if abs(x[i] - round(x[i])) > 1e-6]
            if not fracs:
                continue
            _, bi = max(fracs)
            fv = np.floor(x[bi])
            cv = np.ceil(x[bi])
            row_f = [0.0] * n; row_f[bi] = 1.0
            row_c = [0.0] * n; row_c[bi] = -1.0
            stack.append((ex_A + [row_f], ex_b + [fv]))
            stack.append((ex_A + [row_c], ex_b + [-cv]))

        if best[0] is None:
            return {'status': 'infeasible', 'type': 'Infeasible (ILP)',
                    'optimal_solution': None, 'optimal_value': None,
                    'message': 'No integer solution found'}
        x_int = np.array([round(best[0][i]) for i in range(n)], dtype=float)
        sol_str = ", ".join(f"{self.variable_names[i]}={x_int[i]:.0f}" for i in range(n))
        return {'status': 'success', 'type': 'Integer Optimal',
                'optimal_solution': x_int, 'optimal_value': best[1],
                'message': f'Integer optimal solution found at ({sol_str})'}

    def _classify_result(self, result):
        if result.status == 0:
            optimal_value = -result.fun if self.is_maximize else result.fun
            sol_str = ", ".join(f"{self.variable_names[i]}={result.x[i]:.4f}"
                                for i in range(self.num_variables))
            return {'status': 'success', 'type': 'Finitely Optimal',
                    'optimal_solution': result.x, 'optimal_value': optimal_value,
                    'message': f'Optimal solution found at ({sol_str})'}
        elif result.status == 2:
            return {'status': 'infeasible', 'type': 'Infeasible',
                    'optimal_solution': None, 'optimal_value': None,
                    'message': 'No solution satisfies all constraints'}
        elif result.status == 3:
            return {'status': 'unbounded', 'type': 'Unbounded',
                    'optimal_solution': None, 'optimal_value': None,
                    'message': 'Objective can improve indefinitely'}
        else:
            return {'status': 'error', 'type': 'Unknown',
                    'optimal_solution': None, 'optimal_value': None,
                    'message': f'Solver status {result.status}: {result.message}'}


def compute_lp(objective_str, maximize, constraints_str, non_negative=True, integer=False):
    """
    Solve a linear programming problem.
    Returns standard result dict.
    """
    if not HAS_SCIPY:
        msg = 'ERROR: scipy is required for Linear Programming.\nInstall with: pip install scipy'
        return {'text': msg, 'summary_text': msg}

    try:
        obj_coeffs = [float(x) for x in objective_str.strip().split()]
    except ValueError as e:
        return {'text': f'ERROR parsing objective: {e}', 'summary_text': f'ERROR parsing objective: {e}'}

    num_vars = len(obj_coeffs)
    if num_vars < 1 or num_vars > 7:
        return {'text': f'ERROR: Need 1-7 variables, got {num_vars}.',
                'summary_text': f'ERROR: Need 1-7 variables, got {num_vars}.'}

    solver = LinearProgramSolver(num_vars)
    solver.set_objective(obj_coeffs, maximize)

    if not non_negative:
        solver.bounds = [(None, None)] * num_vars

    constraint_lines = [l.strip() for l in constraints_str.strip().splitlines() if l.strip()]
    for line in constraint_lines:
        try:
            parsed = solver.parse_constraint(line)
            if parsed:
                coeffs, ineq, bound = parsed
                solver.add_constraint(coeffs, ineq, bound)
        except ValueError as e:
            return {'text': f'ERROR in constraint "{line}": {e}',
                    'summary_text': f'ERROR in constraint "{line}": {e}'}

    if not solver.constraints:
        return {'text': 'ERROR: No constraints defined.',
                'summary_text': 'ERROR: No constraints defined.'}

    result = solver.solve(integer=integer)

    # Build output text
    lines = []
    lines.append("LINEAR PROGRAMMING PROBLEM")
    lines.append("=" * 50)

    opt_type = "MAXIMIZE" if maximize else "MINIMIZE"
    obj_terms = []
    for i, coeff in enumerate(obj_coeffs):
        if abs(coeff) > 1e-10:
            sign = '+' if coeff >= 0 and obj_terms else ''
            if coeff < 0:
                sign = ''
            elif not obj_terms:
                sign = ''
            obj_terms.append(f"{sign}{coeff:g}*x{i + 1}")
    lines.append(f"{opt_type}: z = {' '.join(obj_terms)}")

    lines.append("")
    lines.append("SUBJECT TO:")
    for i, (coeffs, bound, ineq) in enumerate(solver.constraints, 1):
        terms = []
        for j, coeff in enumerate(coeffs):
            if abs(coeff) > 1e-10:
                sign = '+' if coeff >= 0 and terms else ''
                if coeff < 0:
                    sign = ''
                elif not terms:
                    sign = ''
                terms.append(f"{sign}{coeff:g}*x{j + 1}")
        lines.append(f"  {' '.join(terms)} {ineq} {bound:g}")
    if non_negative:
        lines.append(f"  All variables >= 0")

    lines.append("")
    lines.append("-" * 50)
    lines.append("SOLUTION")
    lines.append("-" * 50)
    lines.append(f"Status: {result['status'].upper()}")
    lines.append(f"Type: {result['type']}")
    lines.append(f"Message: {result['message']}")

    if result['optimal_solution'] is not None:
        lines.append("")
        lines.append("Optimal Solution:")
        for i in range(num_vars):
            val = result['optimal_solution'][i]
            if integer:
                lines.append(f"  x{i + 1} = {val:.0f}")
            else:
                lines.append(f"  x{i + 1} = {val:.6f}")
        opt_label = "MAXIMUM" if maximize else "MINIMUM"
        if integer:
            lines.append(f"\n  {opt_label} VALUE (z) = {result['optimal_value']:.0f}")
        else:
            lines.append(f"\n  {opt_label} VALUE (z) = {result['optimal_value']:.6f}")

    lines.append("=" * 50)
    text = "\n".join(lines)
    return {'text': text, 'summary_text': text}
