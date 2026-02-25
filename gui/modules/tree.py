import tkinter as tk
from tkinter import ttk

from gui.constants import MAIN_BG
from gui.base_frame import ModuleFrame
from gui.compute.tree import compute_tree


class DecisionTreeFrame(ModuleFrame):
    def __init__(self, parent):
        super().__init__(parent, "Decision Tree & Random Forest")

        f = self.input_frame

        self.X_grid = self.add_matrix_grid(f, "X", rows=5, cols=1,
                                           row_label="samples", col_label="features")
        self.y_grid = self.add_matrix_grid(f, "y", rows=5, cols=1,
                                           row_label="samples", col_label="outputs",
                                           hide_rows=True)
        self._x_rows_syncing = False
        self.X_grid.rows_var.trace_add("write", self._on_X_rows_var_change)

        row1 = tk.Frame(f, bg=MAIN_BG)
        row1.pack(fill=tk.X, pady=4)
        self.task_var = self.add_combo(row1, "Task", ["classification", "regression"], "regression")
        self.crit_var = self.add_combo(row1, "Criterion", ["gini", "entropy"], "gini")
        self.mode_var = self.add_combo(row1, "Mode", ["root", "tree", "forest"], "root")

        row2 = tk.Frame(f, bg=MAIN_BG)
        row2.pack(fill=tk.X, pady=4)
        self.depth_var = self.add_entry(row2, "Max Depth", "3", width=6)
        self.mins_var = self.add_entry(row2, "Min Samples", "2", width=6)
        self.ntrees_var = self.add_entry(row2, "Num Trees (RF)", "10", width=6)
        self.feat_var = self.add_combo(row2, "Max Features", ["sqrt", "log2", "all"], "sqrt")

        self.thr_grid = self.add_matrix_grid(f, "Thresholds (optional, leave 0 = auto)",
                                             rows=1, cols=2,
                                             row_label="features", col_label="thresholds",
                                             hide_rows=True)
        self.X_grid.cols_var.trace_add("write", self._on_X_cols_var_change)

        self.task_var._combobox.bind("<<ComboboxSelected>>", lambda e: self._on_mode_change())
        self.mode_var._combobox.bind("<<ComboboxSelected>>", lambda e: self._on_mode_change())
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

    def _on_X_cols_var_change(self, *args):
        try:
            n_features = int(self.X_grid.cols_var.get())
        except (ValueError, tk.TclError):
            return
        n_features = max(1, min(50, n_features))
        if n_features != self.thr_grid.n_rows:
            self.thr_grid._resize(n_features, self.thr_grid.n_cols)

    def _thr_grid_to_str(self):
        parts = []
        for r in range(self.thr_grid.n_rows):
            vals = []
            for c in range(self.thr_grid.n_cols):
                v = self.thr_grid.cells[r][c].get().strip()
                if v and v != "0":
                    vals.append(v)
            if vals:
                parts.append(f"{r}: " + " ".join(vals))
        return ", ".join(parts)

    def _on_mode_change(self):
        task = self.task_var.get()
        mode = self.mode_var.get()

        self._toggle(self.crit_var._frame, task == "classification", side=tk.LEFT, padx=(0, 10), pady=2)
        self._toggle(self.depth_var._frame, mode != "root", side=tk.LEFT, padx=(0, 10), pady=2)
        self._toggle(self.mins_var._frame, mode != "root", side=tk.LEFT, padx=(0, 10), pady=2)
        self._toggle(self.ntrees_var._frame, mode == "forest", side=tk.LEFT, padx=(0, 10), pady=2)
        self._toggle(self.feat_var._frame, mode == "forest", side=tk.LEFT, padx=(0, 10), pady=2)

    def run(self):
        try:
            result = compute_tree(
                X_str=self.X_grid.get_matrix_string(),
                y_str=self.y_grid.get_matrix_string(),
                task=self.task_var.get(),
                criterion=self.crit_var.get(),
                thr_str=self._thr_grid_to_str(),
                tree_mode=self.mode_var.get(),
                depth=int(self.depth_var.get()),
                min_samples=int(self.mins_var.get()),
                n_trees=int(self.ntrees_var.get()),
                feat_mode=self.feat_var.get(),
            )
            self.show_result(result)
        except Exception as e:
            self.show_output(f"ERROR: {e}")
