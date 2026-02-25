import tkinter as tk
from tkinter import ttk
import numpy as np

from gui.constants import MAIN_BG, ACCENT, FONT_FAMILY
from gui.widgets import RoundedButton
from gui.base_frame import ModuleFrame
from gui.compute.classification import compute_classification


class ClassificationFrame(ModuleFrame):
    def __init__(self, parent):
        super().__init__(parent, "Classification (Logistic Regression)")

        f = self.input_frame
        self.X_grid = self.add_matrix_grid(f, "X (feature matrix)", rows=3, cols=2,
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
        self.y_grid = self.add_matrix_grid(f, "y (labels)", rows=3, cols=1,
                                           row_label="samples", col_label="classes",
                                           hide_rows=True)
        self._x_rows_syncing = False
        self.X_grid.rows_var.trace_add("write", self._on_X_rows_var_change)

        self._row1 = tk.Frame(f, bg=MAIN_BG)
        self._row1.pack(fill=tk.X, pady=4)
        self.lr_var = self.add_entry(self._row1, "Learning Rate", "0.1")
        self.iters_var = self.add_entry(self._row1, "Epochs", "300")
        self.l2_var = self.add_entry(self._row1, "L2 Lambda", "0.0", width=6)
        self.threshold_var = self.add_entry(self._row1, "Threshold", "0.5", width=6)

        self._row2 = tk.Frame(f, bg=MAIN_BG)
        self._row2.pack(fill=tk.X, pady=4)
        self.pen_bias_var = self.add_check(self._row2, "Penalize Bias")
        self.metrics_var = self.add_check(self._row2, "Show Metrics", True)

        # Extra settings (batch, batch size, momentum) - stored as StringVars with defaults
        self.batch_var = tk.StringVar(value="gd")
        self.bs_var = tk.StringVar(value="0")
        self.momentum_var = tk.StringVar(value="0.0")

        self._extra_btn_frame = tk.Frame(f, bg=MAIN_BG)
        self._extra_btn_frame.pack(fill=tk.X, pady=(2, 4))
        self._extra_btn = RoundedButton(self._extra_btn_frame, text="Extra Settings",
                      command=self._open_extra_settings,
                      font=(FONT_FAMILY, 10), padx=10, pady=2,
                      bg_color="#e0e0e0", hover_color="#c8c8c8", press_color="#b0b0b0")
        self._extra_btn.pack(anchor="w")

        self._row3 = tk.Frame(f, bg=MAIN_BG)
        self._row3.pack(fill=tk.X, pady=4)
        self.w_init_var = self.add_button_group(self._row3, "W Init", ["zeros", "manual", "random"], "zeros",
                                                on_change=self._on_mode_change)
        self._train_w_grid = self.add_matrix_grid(f, "W (manual init)", rows=3, cols=1,
                                                   row_label="weights", col_label="classes")

        # --- Predict widgets ---
        self._pred_frame = tk.Frame(f, bg=MAIN_BG)
        self._pred_frame.pack(fill=tk.X, pady=4)
        self.pred_binary_var = self.add_check(self._pred_frame, "Binary")
        self.pred_threshold_var = self.add_entry(self._pred_frame, "Threshold", "0.5", width=6)

        self._pred_w_grid = self.add_matrix_grid(f, "W (trained weights)", rows=3, cols=1,
                                                  row_label="weights", col_label="classes")

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
        self._set_mode("predict")
        self._pred_w_grid.set_from_matrix(np.atleast_2d(W))

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

    def _open_extra_settings(self):
        win = tk.Toplevel(self)
        win.title("Extra Settings \u2013 Classification")
        win.configure(bg=MAIN_BG)
        win.resizable(False, False)
        win.grab_set()
        # Position near the button
        bx = self._extra_btn.winfo_rootx()
        by = self._extra_btn.winfo_rooty() + self._extra_btn.winfo_height()
        win.geometry(f"+{bx}+{by}")

        pad = {'padx': 10, 'pady': 6}

        tk.Label(win, text="Batch Type", bg=MAIN_BG, font=(FONT_FAMILY, 11)).grid(row=0, column=0, sticky="w", **pad)
        batch_cb = ttk.Combobox(win, values=["gd", "sgd", "mb"], state="readonly", width=8)
        batch_cb.set(self.batch_var.get())
        batch_cb.grid(row=0, column=1, sticky="w", **pad)

        tk.Label(win, text="Batch Size", bg=MAIN_BG, font=(FONT_FAMILY, 11)).grid(row=1, column=0, sticky="w", **pad)
        bs_ent = tk.Entry(win, width=8, font=(FONT_FAMILY, 11))
        bs_ent.insert(0, self.bs_var.get())
        bs_ent.grid(row=1, column=1, sticky="w", **pad)

        tk.Label(win, text="Momentum", bg=MAIN_BG, font=(FONT_FAMILY, 11)).grid(row=2, column=0, sticky="w", **pad)
        mom_ent = tk.Entry(win, width=8, font=(FONT_FAMILY, 11))
        mom_ent.insert(0, self.momentum_var.get())
        mom_ent.grid(row=2, column=1, sticky="w", **pad)

        def _save():
            self.batch_var.set(batch_cb.get())
            self.bs_var.set(bs_ent.get())
            self.momentum_var.set(mom_ent.get())
            win.destroy()

        RoundedButton(win, text="OK", command=_save,
                      font=(FONT_FAMILY, 11, "bold"), padx=20, pady=4,
                      bg_color=ACCENT, fg_color="#fff",
                      hover_color="#3a3d5c", press_color="#1a1d32").grid(row=3, column=0, columnspan=2, pady=10)

    def _set_mode(self, mode):
        self._mode_val = mode
        # Update button styles
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
        w_init = self.w_init_var.get()

        # Train-only widgets
        self._toggle(self.y_grid, is_train, fill=tk.X, pady=2)
        self._toggle(self._row1, is_train, fill=tk.X, pady=4)
        self._toggle(self._row2, is_train, fill=tk.X, pady=4)
        self._toggle(self._extra_btn_frame, is_train, fill=tk.X, pady=(2, 4))
        self._toggle(self._row3, is_train, fill=tk.X, pady=4)
        self._toggle(self._train_w_grid, is_train and w_init == "manual", fill=tk.X, pady=2)

        # Predict-only widgets
        self._toggle(self._pred_frame, not is_train, fill=tk.X, pady=4)
        self._toggle(self._pred_w_grid, not is_train, fill=tk.X, pady=2)

    def run(self):
        try:
            if self._mode_val == "train":
                w_str = self._train_w_grid.get_matrix_string()
            else:
                w_str = self._pred_w_grid.get_matrix_string()
            result = compute_classification(
                X_str=self.X_grid.get_matrix_string(),
                y_str=self.y_grid.get_matrix_string(),
                mode_choice=self._mode_val,
                lr=float(self.lr_var.get()),
                iters=int(self.iters_var.get()),
                batch_type=self.batch_var.get(),
                batch_size=int(self.bs_var.get()),
                momentum=float(self.momentum_var.get()),
                l2=float(self.l2_var.get()),
                penalize_bias=self.pen_bias_var.get(),
                threshold=float(self.threshold_var.get()),
                w_init_choice=self.w_init_var.get(),
                w_init_str=w_str,
                pred_binary=self.pred_binary_var.get(),
                pred_W_str=w_str,
                pred_threshold=float(self.pred_threshold_var.get()),
                show_metrics=self.metrics_var.get(),
            )
            self.show_result(result)
        except Exception as e:
            self.show_output(f"ERROR: {e}")
