import tkinter as tk
from tkinter import ttk
import re

from gui.constants import MAIN_BG, ACCENT, FONT_FAMILY
from gui.widgets import RoundedButton
from gui.base_frame import ModuleFrame
from gui.widgets_cost import (HAS_MATPLOTLIB, HAS_SYMPY, _SUBSCRIPTS,
                              ExpressionEditor, MathPreview, MathKeyboard)
from gui.compute.cost_minimizer import compute_cost_minimizer


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

        self._var_hint_label = ttk.Label(expr_fr,
                  text="",
                  foreground="#e06c75", font=(FONT_FAMILY, 8))

        # Live math preview (rendered LaTeX)
        if HAS_MATPLOTLIB and HAS_SYMPY:
            self.math_preview = MathPreview(expr_fr)
            self.math_preview.pack(fill=tk.X, pady=(2, 0))
        else:
            self.math_preview = None

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
        ttk.Label(row1, text="lr:").pack(side=tk.LEFT)
        self.lr_var = tk.StringVar(value="0.1")
        ttk.Entry(row1, textvariable=self.lr_var, width=8).pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(row1, text="Iterations:").pack(side=tk.LEFT)
        self.iters_var = tk.StringVar(value="1")
        ttk.Entry(row1, textvariable=self.iters_var, width=8).pack(side=tk.LEFT, padx=(2, 0))

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
        raw = self.expr_editor.get_expression()
        if self.math_preview is not None:
            self.math_preview.update_expression(raw)

        # Warn if user typed 'x' directly instead of using variable buttons
        typed = self.expr_editor.get_raw_text()
        if re.search(r'x\d?', typed):
            self._var_hint_label.configure(
                text="Don't type variables \u2014 use the blue buttons below to insert x\u2081\u2013x\u2085.")
            self._var_hint_label.pack(anchor="w")
        else:
            self._var_hint_label.pack_forget()

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
