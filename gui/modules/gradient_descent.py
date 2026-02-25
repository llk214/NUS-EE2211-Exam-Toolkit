import tkinter as tk
from tkinter import ttk

from gui.constants import MAIN_BG
from gui.base_frame import ModuleFrame
from gui.compute.gradient_descent import compute_gradient_descent


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
