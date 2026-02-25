import tkinter as tk
from tkinter import ttk

from gui.constants import MAIN_BG, ACCENT, FONT_FAMILY
from gui.widgets import RoundedButton
from gui.base_frame import ModuleFrame
from gui.compute.clustering import compute_clustering


class ClusteringFrame(ModuleFrame):
    def __init__(self, parent):
        super().__init__(parent, "Clustering (K-Means / Fuzzy C-Means)")

        f = self.input_frame
        self.X_grid = self.add_matrix_grid(f, "X (data matrix)", rows=4, cols=1,
                                           row_label="samples", col_label="features")

        row1 = tk.Frame(f, bg=MAIN_BG)
        row1.pack(fill=tk.X, pady=4)
        self.method_var = self.add_button_group(row1, "Method", ["kmeans", "kmeans++", "fcm"], "kmeans",
                                                on_change=self._on_method_change)
        self.K_var = self.add_entry(row1, "K (for K++)", "2", width=6)
        self.fuzz_var = self.add_entry(row1, "Fuzzifier m", "2.0", width=6)

        self.C0_grid = self.add_matrix_grid(f, "C0 (initial centroids)", rows=2, cols=1,
                                            row_label="centroids", col_label="features",
                                            hide_cols=True)
        self._x_cols_syncing = False
        self.X_grid.cols_var.trace_add("write", self._on_X_cols_var_change)

        # Extra settings (Max Iter, Tol) - stored as StringVars with defaults
        self.maxiter_var = tk.StringVar(value="200")
        self.tol_var = tk.StringVar(value="1e-4")

        extra_btn_frame = tk.Frame(f, bg=MAIN_BG)
        extra_btn_frame.pack(fill=tk.X, pady=(2, 4))
        self._extra_btn = RoundedButton(extra_btn_frame, text="Extra Settings",
                      command=self._open_extra_settings,
                      font=(FONT_FAMILY, 10), padx=10, pady=2,
                      bg_color="#e0e0e0", hover_color="#c8c8c8", press_color="#b0b0b0")
        self._extra_btn.pack(anchor="w")

        self._on_method_change()

    def _on_X_cols_var_change(self, *args):
        if self._x_cols_syncing:
            return
        try:
            new_cols = int(self.X_grid.cols_var.get())
        except (ValueError, tk.TclError):
            return
        new_cols = max(1, min(50, new_cols))
        if new_cols != self.C0_grid.n_cols:
            self._x_cols_syncing = True
            self.C0_grid._resize(self.C0_grid.n_rows, new_cols)
            self._x_cols_syncing = False

    def _on_method_change(self):
        method = self.method_var.get()
        self._toggle(self.C0_grid, method != "kmeans++", fill=tk.X, pady=2)
        self._toggle(self.K_var._frame, method == "kmeans++", side=tk.LEFT, padx=(0, 10), pady=2)
        self._toggle(self.fuzz_var._frame, method == "fcm", side=tk.LEFT, padx=(0, 10), pady=2)

    def _open_extra_settings(self):
        win = tk.Toplevel(self)
        win.title("Extra Settings \u2013 Clustering")
        win.configure(bg=MAIN_BG)
        win.resizable(False, False)
        win.grab_set()
        bx = self._extra_btn.winfo_rootx()
        by = self._extra_btn.winfo_rooty() + self._extra_btn.winfo_height()
        win.geometry(f"+{bx}+{by}")

        pad = {'padx': 10, 'pady': 6}

        tk.Label(win, text="Max Iter", bg=MAIN_BG, font=(FONT_FAMILY, 11)).grid(row=0, column=0, sticky="w", **pad)
        maxiter_ent = tk.Entry(win, width=8, font=(FONT_FAMILY, 11))
        maxiter_ent.insert(0, self.maxiter_var.get())
        maxiter_ent.grid(row=0, column=1, sticky="w", **pad)

        tk.Label(win, text="Tol", bg=MAIN_BG, font=(FONT_FAMILY, 11)).grid(row=1, column=0, sticky="w", **pad)
        tol_ent = tk.Entry(win, width=10, font=(FONT_FAMILY, 11))
        tol_ent.insert(0, self.tol_var.get())
        tol_ent.grid(row=1, column=1, sticky="w", **pad)

        def _save():
            self.maxiter_var.set(maxiter_ent.get())
            self.tol_var.set(tol_ent.get())
            win.destroy()

        RoundedButton(win, text="OK", command=_save,
                      font=(FONT_FAMILY, 11, "bold"), padx=20, pady=4,
                      bg_color=ACCENT, fg_color="#fff",
                      hover_color="#3a3d5c", press_color="#1a1d32").grid(row=2, column=0, columnspan=2, pady=10)

    def run(self):
        try:
            result = compute_clustering(
                X_str=self.X_grid.get_matrix_string(),
                method=self.method_var.get(),
                C0_str=self.C0_grid.get_matrix_string(),
                K_val=int(self.K_var.get()),
                fuzzifier=float(self.fuzz_var.get()),
                max_iter=int(self.maxiter_var.get()),
                tol=float(self.tol_var.get()),
            )
            self.show_result(result)
        except Exception as e:
            self.show_output(f"ERROR: {e}")
