import tkinter as tk
from tkinter import ttk

from gui.constants import MAIN_BG, FONT_FAMILY
from gui.widgets import RoundedButton, MatrixGrid
from gui.base_frame import ModuleFrame
from gui.compute.neural import compute_neural


class NeuralNetFrame(ModuleFrame):
    def __init__(self, parent):
        super().__init__(parent, "Neural Network (MLP)")

        f = self.input_frame
        self.X_grid = self.add_matrix_grid(f, "X (NO bias)", rows=3, cols=2,
                                           row_label="samples", col_label="features")
        self.Y_grid = self.add_matrix_grid(f, "Y (targets)", rows=3, cols=2,
                                           row_label="samples", col_label="outputs",
                                           hide_rows=True)
        # Auto-sync Y rows from X
        self._x_rows_syncing = False
        self.X_grid.rows_var.trace_add("write", self._on_X_rows_var_change)

        # Layer controls
        layer_header = tk.Frame(f, bg=MAIN_BG)
        layer_header.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(layer_header, text="Layers", style="Header.TLabel").pack(side=tk.LEFT)
        RoundedButton(layer_header, text="+ Add Layer", command=self.add_layer
                      ).pack(side=tk.LEFT, padx=8)

        self.layers_frame = tk.Frame(f, bg=MAIN_BG)
        self.layers_frame.pack(fill=tk.X)
        self.layer_widgets = []
        self.add_layer()  # default 1 layer

        # Auto-sync W grid dimensions when X cols change
        self.X_grid.cols_var.trace_add("write", lambda *a: self._sync_all_W_dims())

        # Hyperparams
        row = tk.Frame(f, bg=MAIN_BG)
        row.pack(fill=tk.X, pady=8)
        self.lr_var = self.add_entry(row, "lr", "0.1")
        self.iters_var = self.add_entry(row, "Iterations", "1")
        self.loss_var = self.add_combo(row, "Loss", ["mse", "ce"], "mse")

    def add_layer(self):
        idx = len(self.layer_widgets)
        fr = tk.LabelFrame(self.layers_frame, text=f"Layer {idx + 1}", bg=MAIN_BG,
                           font=(FONT_FAMILY, 9, "bold"), padx=6, pady=4)
        fr.pack(fill=tk.X, pady=2)

        row = tk.Frame(fr, bg=MAIN_BG)
        row.pack(fill=tk.X)

        ttk.Label(row, text="Neurons:").pack(side=tk.LEFT)
        neurons_var = tk.StringVar(value="2")
        ttk.Entry(row, textvariable=neurons_var, width=5).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(row, text="Activation:").pack(side=tk.LEFT)
        act_var = tk.StringVar(value="relu")
        ttk.Combobox(row, textvariable=act_var, values=["relu", "sigmoid", "linear", "softmax", "tanh"],
                     width=8, state="readonly").pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(row, text="Init:").pack(side=tk.LEFT)
        init_var = tk.StringVar(value="xavier")
        ttk.Combobox(row, textvariable=init_var, values=["zeros", "xavier", "he", "random", "manual"],
                     width=8, state="readonly").pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(row, text="Seed:").pack(side=tk.LEFT)
        seed_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=seed_var, width=5).pack(side=tk.LEFT, padx=(2, 10))

        # Manual W input (rows and cols auto-synced)
        w_grid = MatrixGrid(fr, "Manual W (incl bias row)", rows=3, cols=2,
                            row_label="inputs+bias", col_label="neurons",
                            hide_rows=True, hide_cols=True)
        w_grid.pack(fill=tk.X, pady=2)

        # Remove button at the bottom of this layer
        remove_btn = RoundedButton(fr, text="Remove this layer",
                                   command=lambda: self.remove_layer(lw),
                                   font=(FONT_FAMILY, 9), bg_color="#e8c0c0",
                                   hover_color="#d9a0a0", press_color="#c88080")
        remove_btn.pack(anchor="e", pady=(2, 0))

        lw = {
            'frame': fr, 'neurons': neurons_var, 'activation': act_var,
            'init': init_var, 'seed': seed_var, 'W_manual': w_grid
        }
        self.layer_widgets.append(lw)

        # Sync W cols when neurons changes, and sync all W rows (chain effect)
        neurons_var.trace_add("write", lambda *args, l=lw: self._on_neurons_change(l))
        init_var.trace_add("write", lambda *args, l=lw: self._on_init_change(l))
        self._sync_all_W_dims()
        self._on_init_change(lw)

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

    def _on_neurons_change(self, lw):
        """When a layer's neuron count changes, sync its W cols and re-sync all W rows."""
        if lw not in self.layer_widgets:
            return
        try:
            n = int(lw['neurons'].get())
        except (ValueError, tk.TclError):
            return
        n = max(1, min(50, n))
        w_grid = lw['W_manual']
        if w_grid.n_cols != n:
            w_grid._resize(w_grid.n_rows, n)
        # Changing neurons affects the NEXT layer's W rows
        self._sync_all_W_dims()

    def _sync_all_W_dims(self):
        """Sync W grid rows for all layers based on the chain: X cols -> layer neurons."""
        try:
            prev_dim = int(self.X_grid.cols_var.get())
        except (ValueError, tk.TclError):
            return
        for lw in self.layer_widgets:
            w_grid = lw['W_manual']
            expected_rows = prev_dim + 1  # +1 for bias
            try:
                n = int(lw['neurons'].get())
            except (ValueError, tk.TclError):
                n = w_grid.n_cols
            n = max(1, min(50, n))
            # Sync cols to neurons
            if w_grid.n_cols != n:
                w_grid._resize(w_grid.n_rows, n)
            # Sync rows to prev_dim + 1
            if w_grid.n_rows != expected_rows:
                w_grid._resize(expected_rows, w_grid.n_cols)
            prev_dim = n

    def _on_init_change(self, lw):
        if lw in self.layer_widgets:
            is_manual = lw['init'].get() == "manual"
            self._toggle(lw['W_manual'], is_manual, fill=tk.X, pady=2)

    def remove_layer(self, lw):
        if len(self.layer_widgets) <= 1:
            return
        self.layer_widgets.remove(lw)
        lw['frame'].destroy()
        # Renumber remaining layers
        for i, layer in enumerate(self.layer_widgets):
            layer['frame'].configure(text=f"Layer {i + 1}")
        # Re-sync W dimensions after removal
        self._sync_all_W_dims()

    def run(self):
        try:
            layer_configs = []
            for lw in self.layer_widgets:
                w_str = lw['W_manual'].get_matrix_string()
                # Check if all cells are just "0" (no manual input)
                all_zero = all(v.strip() in ("0", "0.0", "") for v in w_str.replace(",", " ").split())
                layer_configs.append({
                    'neurons': int(lw['neurons'].get()),
                    'activation': lw['activation'].get(),
                    'init': lw['init'].get(),
                    'seed': lw['seed'].get(),
                    'W_manual': w_str if not all_zero else None,
                })

            result = compute_neural(
                X_str=self.X_grid.get_matrix_string(),
                Y_str=self.Y_grid.get_matrix_string(),
                layer_configs=layer_configs,
                lr=float(self.lr_var.get()),
                iters=int(self.iters_var.get()),
                loss_type=self.loss_var.get(),
            )
            self.show_result(result)
        except Exception as e:
            self.show_output(f"ERROR: {e}")
