import tkinter as tk
from tkinter import ttk
import numpy as np

from gui.constants import MAIN_BG, ACCENT, FONT_FAMILY
from gui.widgets import RoundedButton
from gui.base_frame import ModuleFrame
from gui.compute.regression import compute_regression, compute_regression_predict


class RegressionFrame(ModuleFrame):
    def __init__(self, parent):
        super().__init__(parent, "Regression (OLS / Ridge / Polynomial)")

        f = self.input_frame
        self.X_grid = self.add_matrix_grid(f, "X", rows=5, cols=1,
                                           row_label="samples", col_label="features")

        # Mode toggle buttons (Train / Predict)
        mode_frame = tk.Frame(f, bg=MAIN_BG)
        mode_frame.pack(fill=tk.X, pady=(4, 4))
        self._mode_val = "train"
        self._train_btn = RoundedButton(mode_frame, text="Train", command=lambda: self._set_mode("train"),
                                        font=(FONT_FAMILY, 11, "bold"), padx=16, pady=3,
                                        bg_color=ACCENT, fg_color="#fff",
                                        hover_color="#3a3d5c", press_color="#1a1d32")
        self._train_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._predict_btn = RoundedButton(mode_frame, text="Predict", command=lambda: self._set_mode("predict"),
                                          font=(FONT_FAMILY, 11, "bold"), padx=16, pady=3,
                                          bg_color="#e0e0e0", fg_color="#222",
                                          hover_color="#c8c8c8", press_color="#b0b0b0")
        self._predict_btn.pack(side=tk.LEFT)

        # --- Train widgets ---
        self.Y_grid = self.add_matrix_grid(f, "Y", rows=5, cols=1,
                                           row_label="samples", col_label="outputs",
                                           hide_rows=True)
        self._x_rows_syncing = False
        self.X_grid.rows_var.trace_add("write", self._on_X_rows_var_change)

        self._train_row = tk.Frame(f, bg=MAIN_BG)
        self._train_row.pack(fill=tk.X, pady=4)
        self.model_var = self.add_button_group(self._train_row, "Model", ["ols", "ridge", "polynomial"], "ols",
                                                on_change=self._on_mode_change)
        self.alpha_var = self.add_entry(self._train_row, "Alpha (Ridge)", "1.0")
        self.degree_var = self.add_entry(self._train_row, "Degree (Poly)", "2", width=6)
        self.pen_bias_var = self.add_check(self._train_row, "Penalize Bias")

        # --- Predict widgets ---
        self._pred_row = tk.Frame(f, bg=MAIN_BG)
        self._pred_row.pack(fill=tk.X, pady=4)
        self.pred_model_var = self.add_button_group(self._pred_row, "Model", ["ols/ridge", "polynomial"], "ols/ridge",
                                                    on_change=self._on_mode_change)
        self.pred_degree_var = self.add_entry(self._pred_row, "Degree (Poly)", "2", width=6)

        self.W_grid = self.add_matrix_grid(f, "W (trained weights)", rows=1, cols=1,
                                           row_label="features", col_label="outputs")
        self.b_grid = self.add_matrix_grid(f, "b (intercept)", rows=1, cols=1,
                                           row_label="row", col_label="outputs")

        # "Use for Predict" button in button bar
        self._use_predict_btn = RoundedButton(self._btn_frame, text="Use for Predict",
                      command=self._use_for_predict,
                      font=(FONT_FAMILY, 10), padx=10, pady=2,
                      bg_color="#a6e3a1", fg_color="#1e1e2e",
                      hover_color="#94d990", press_color="#7cc97a")
        self._use_predict_btn.pack(side=tk.LEFT, padx=(8, 0))

        self._set_mode("train")

    def _use_for_predict(self):
        if self._last_result is None or self._last_result.get('weights') is None:
            return
        W = self._last_result['weights']
        b = self._last_result.get('bias')
        self._set_mode("predict")
        self.W_grid.set_from_matrix(np.atleast_2d(W))
        if b is not None:
            self.b_grid.set_from_matrix(np.atleast_2d(np.atleast_1d(b)))

    def _on_X_rows_var_change(self, *args):
        if self._x_rows_syncing:
            return
        try:
            new_rows = int(self.X_grid.rows_var.get())
        except (ValueError, tk.TclError):
            return
        new_rows = max(1, min(50, new_rows))
        if new_rows != self.Y_grid.n_rows:
            self._x_rows_syncing = True
            self.Y_grid._resize(new_rows, self.Y_grid.n_cols)
            self._x_rows_syncing = False

    def _set_mode(self, mode):
        self._mode_val = mode
        if mode == "train":
            self._train_btn._bg_color = ACCENT
            self._train_btn._fg_color = "#fff"
            self._train_btn._hover_color = "#3a3d5c"
            self._train_btn._draw(ACCENT)
            self._predict_btn._bg_color = "#e0e0e0"
            self._predict_btn._fg_color = "#222"
            self._predict_btn._hover_color = "#c8c8c8"
            self._predict_btn._draw("#e0e0e0")
        else:
            self._predict_btn._bg_color = ACCENT
            self._predict_btn._fg_color = "#fff"
            self._predict_btn._hover_color = "#3a3d5c"
            self._predict_btn._draw(ACCENT)
            self._train_btn._bg_color = "#e0e0e0"
            self._train_btn._fg_color = "#222"
            self._train_btn._hover_color = "#c8c8c8"
            self._train_btn._draw("#e0e0e0")
        self._on_mode_change()

    def _on_mode_change(self):
        is_train = (self._mode_val == "train")
        model = self.model_var.get()
        pred_model = self.pred_model_var.get()

        # Train widgets
        self._toggle(self.Y_grid, is_train, fill=tk.X, pady=2)
        self._toggle(self._train_row, is_train, fill=tk.X, pady=4)
        if is_train:
            self._toggle(self.alpha_var._frame, model != "ols", side=tk.LEFT, padx=(0, 10), pady=2)
            self._toggle(self.degree_var._frame, model == "polynomial", side=tk.LEFT, padx=(0, 10), pady=2)
            self._toggle(self.pen_bias_var._frame, model != "ols", side=tk.LEFT, padx=(0, 10), pady=2)

        # Predict widgets
        self._toggle(self._pred_row, not is_train, fill=tk.X, pady=4)
        self._toggle(self.W_grid, not is_train, fill=tk.X, pady=2)
        self._toggle(self.b_grid, not is_train, fill=tk.X, pady=2)
        if not is_train:
            self._toggle(self.pred_degree_var._frame, pred_model == "polynomial",
                         side=tk.LEFT, padx=(0, 10), pady=2)

    def run(self):
        try:
            if self._mode_val == "train":
                result = compute_regression(
                    X_str=self.X_grid.get_matrix_string(),
                    Y_str=self.Y_grid.get_matrix_string(),
                    model=self.model_var.get(),
                    alpha=float(self.alpha_var.get()),
                    degree=int(self.degree_var.get()),
                    penalize_bias=self.pen_bias_var.get(),
                )
            else:
                model = "polynomial" if self.pred_model_var.get() == "polynomial" else "ols"
                result = compute_regression_predict(
                    X_str=self.X_grid.get_matrix_string(),
                    W_str=self.W_grid.get_matrix_string(),
                    b_str=self.b_grid.get_matrix_string(),
                    model=model,
                    degree=int(self.pred_degree_var.get()),
                )
            self.show_result(result)
        except Exception as e:
            self.show_output(f"ERROR: {e}")
