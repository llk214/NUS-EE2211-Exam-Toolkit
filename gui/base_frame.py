import tkinter as tk
from tkinter import ttk
import numpy as np

from gui.constants import MAIN_BG, ACCENT, FONT_FAMILY, MONO_FONT
from gui.widgets import RoundedButton, MatrixGrid, CollapsibleSection, TreeVisualizer, ToggleSwitch


class ModuleFrame(tk.Frame):
    """Base class for all module frames."""

    def __init__(self, parent, title):
        super().__init__(parent, bg=MAIN_BG)
        self._last_result = None
        self.title_label = ttk.Label(self, text=title, style="Header.TLabel")
        self.title_label.pack(anchor="w", pady=(0, 8))

        # Paned window: top = inputs, bottom = output
        self.pane = ttk.PanedWindow(self, orient=tk.VERTICAL)
        self.pane.pack(fill=tk.BOTH, expand=True)

        # Input area (scrollable, no visible scrollbar)
        self.input_outer = tk.Frame(self.pane, bg=MAIN_BG)
        self.input_canvas = tk.Canvas(self.input_outer, bg=MAIN_BG, highlightthickness=0)
        self.input_frame = tk.Frame(self.input_canvas, bg=MAIN_BG)

        self.input_frame.bind("<Configure>", lambda e: self.input_canvas.configure(
            scrollregion=self.input_canvas.bbox("all")))
        self.canvas_window = self.input_canvas.create_window((0, 0), window=self.input_frame, anchor="nw")
        self.input_canvas.bind("<Configure>", lambda e: self.input_canvas.itemconfig(
            self.canvas_window, width=e.width))

        self.input_canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mousewheel to input canvas
        self.input_frame.bind("<Enter>", lambda e: self._bind_mousewheel(self.input_canvas))
        self.input_frame.bind("<Leave>", lambda e: self._unbind_mousewheel(self.input_canvas))

        self.pane.add(self.input_outer, weight=1)

        # Button bar
        self._btn_frame = tk.Frame(self, bg=MAIN_BG)
        self._btn_frame.pack(fill=tk.X, pady=4)
        RoundedButton(self._btn_frame, text="Run", command=self.run,
                      font=(FONT_FAMILY, 12, "bold"), bg_color=ACCENT, fg_color="#fff",
                      hover_color="#3a3d5c", press_color="#1a1d32",
                      padx=14, pady=4).pack(side=tk.LEFT, padx=(0, 8))
        RoundedButton(self._btn_frame, text="Clear Output", command=self.clear_output
                      ).pack(side=tk.LEFT)

        # Output area with tabbed notebook
        self.output_frame = tk.Frame(self, bg=MAIN_BG)
        self.output_frame.pack(fill=tk.BOTH, expand=True)

        # Style the notebook tabs to match the dark theme
        style = ttk.Style()
        style.configure("Dark.TNotebook", background=MAIN_BG)
        style.configure("Dark.TNotebook.Tab", background="#2a2a3e", foreground="#cdd6f4",
                        font=("Consolas", 10), padding=[10, 4])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", "#1e1e2e"), ("!selected", "#2a2a3e")],
                  foreground=[("selected", "#89b4fa"), ("!selected", "#cdd6f4")])

        self.output_notebook = ttk.Notebook(self.output_frame, style="Dark.TNotebook")
        self.output_notebook.pack(fill=tk.BOTH, expand=True)

        # Summary tab (always present)
        self.summary_tab = tk.Text(self.output_notebook, font=MONO_FONT, wrap=tk.WORD, height=12,
                                   bg="#1e1e2e", fg="#cdd6f4", insertbackground="#cdd6f4",
                                   selectbackground="#45475a", relief=tk.FLAT, padx=8, pady=6)
        self.output_notebook.add(self.summary_tab, text="Summary")

        # Keep reference to old output for backward compat
        self.output = self.summary_tab

        # Bind mousewheel to summary
        self.summary_tab.bind("<Enter>", lambda e: self._bind_mousewheel(self.summary_tab))
        self.summary_tab.bind("<Leave>", lambda e: self._unbind_mousewheel(self.summary_tab))

        # Iterations tab (created on demand)
        self._iterations_tab = None
        self._iterations_canvas = None
        self._iterations_inner = None

        # Weights tab (created on demand)
        self._weights_tab = None

        # Tree tab (created on demand)
        self._tree_tab = None

        # --- Hover-based output resize ---
        self._hover_timer = None
        self._output_state = "collapsed"
        self._INPUT_MIN_HEIGHT = 120      # input height when output is expanded
        self._SUMMARY_HEIGHT_COLLAPSED = 1   # lines when collapsed
        self._SUMMARY_HEIGHT_EXPANDED = 12   # lines when expanded
        self._HOVER_DELAY = 500           # ms before expanding

        # Bind hover on the output area
        self.output_frame.bind("<Enter>", self._on_output_enter)
        self.output_frame.bind("<Leave>", self._on_output_leave)
        self.output_notebook.bind("<Enter>", self._on_output_enter)

        # Bind hover on input area to collapse output
        self.input_outer.bind("<Enter>", self._on_input_enter)
        self.input_canvas.bind("<Enter>", self._on_input_enter)
        self._btn_frame.bind("<Enter>", self._on_input_enter)

        # Start collapsed — summary text is 1 line, input gets the space
        self._collapse_output()

    def _has_output(self):
        """Check if there is any output content."""
        content = self.summary_tab.get("1.0", tk.END).strip()
        return len(content) > 0

    def _on_output_enter(self, event=None):
        """Mouse entered output area — schedule expansion after delay."""
        if self._output_state == "expanded" or not self._has_output():
            return
        if self._hover_timer is not None:
            self.after_cancel(self._hover_timer)
        self._hover_timer = self.after(self._HOVER_DELAY, self._expand_output)

    def _on_output_leave(self, event=None):
        """Mouse left output area — cancel pending expansion."""
        if self._hover_timer is not None:
            self.after_cancel(self._hover_timer)
            self._hover_timer = None

    def _on_input_enter(self, event=None):
        """Mouse entered input area — collapse output to ~150px."""
        if self._hover_timer is not None:
            self.after_cancel(self._hover_timer)
            self._hover_timer = None
        if self._output_state == "expanded" or (self._output_state == "default" and self._has_output()):
            self._collapse_output()

    def _expand_output(self):
        """Shrink input pane, grow output."""
        self._hover_timer = None
        if self._output_state == "expanded":
            return
        self._output_state = "expanded"
        self.summary_tab.configure(height=self._SUMMARY_HEIGHT_EXPANDED)
        self.pane.configure(height=self._INPUT_MIN_HEIGHT)
        self.pane.pack_configure(expand=False)
        self.output_frame.pack_configure(expand=True)

    def _collapse_output(self):
        """Shrink output to 1 line, give input all space."""
        self._output_state = "collapsed"
        self.summary_tab.configure(height=self._SUMMARY_HEIGHT_COLLAPSED)
        self.pane.configure(height=0)
        self.pane.pack_configure(expand=True)
        self.output_frame.pack_configure(expand=False)

    def _reset_layout(self):
        """Restore default layout (output small, input gets space)."""
        if self._hover_timer is not None:
            self.after_cancel(self._hover_timer)
            self._hover_timer = None
        self._collapse_output()

    def _bind_mousewheel(self, widget):
        widget.bind_all("<MouseWheel>", lambda e: self._on_mousewheel(e, widget))

    def _unbind_mousewheel(self, widget):
        widget.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event, widget):
        if isinstance(widget, tk.Canvas):
            widget.yview_scroll(int(-1 * (event.delta / 120)), "units")
        else:
            widget.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _remove_tab(self, tab_widget):
        """Remove a tab from the notebook if it exists."""
        if tab_widget is not None:
            try:
                self.output_notebook.forget(tab_widget)
            except tk.TclError:
                pass

    def _build_iterations_tab(self, iterations):
        """Build or rebuild the Iterations tab with collapsible sections."""
        self._remove_tab(self._iterations_tab)

        outer = tk.Frame(self.output_notebook, bg="#1e1e2e")

        # Scrollable area
        canvas = tk.Canvas(outer, bg="#1e1e2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg="#1e1e2e")

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_win, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mousewheel binding
        inner.bind("<Enter>", lambda e: self._bind_mousewheel(canvas))
        inner.bind("<Leave>", lambda e: self._unbind_mousewheel(canvas))

        # Determine which iterations to show
        n = len(iterations)
        show_all_ref = [False]
        max_collapsed = 50

        def _populate(show_all=False):
            for w in inner.winfo_children():
                w.destroy()

            if n <= max_collapsed or show_all:
                items = iterations
            else:
                items = iterations[:10] + [None] + iterations[-10:]

            for item in items:
                if item is None:
                    # "Show all" button
                    btn_frame = tk.Frame(inner, bg="#1e1e2e")
                    btn_frame.pack(fill=tk.X, pady=4, padx=8)
                    RoundedButton(btn_frame, text=f"Show all {n} iterations",
                                  command=lambda: _populate(True),
                                  font=(FONT_FAMILY, 10), padx=12, pady=3,
                                  bg_color="#45475a", fg_color="#cdd6f4",
                                  hover_color="#585b70", press_color="#6c7086"
                                  ).pack(anchor="w")
                    continue

                loss_str = f" | loss={item['loss']:.6f}" if 'loss' in item else ""
                title = f"Iteration {item['iter']}{loss_str}"
                section = CollapsibleSection(inner, title, item['text'], expanded=False)
                section.pack(fill=tk.X, padx=4, pady=1)

        _populate()

        self._iterations_tab = outer
        self._iterations_canvas = canvas
        self._iterations_inner = inner
        self.output_notebook.add(outer, text="Iterations")

    def _build_weights_tab(self, result):
        """Build or rebuild the Weights tab showing weight matrices."""
        self._remove_tab(self._weights_tab)

        outer = tk.Frame(self.output_notebook, bg="#1e1e2e")
        weights = result.get('weights')
        bias = result.get('bias')

        if weights is None:
            self._weights_tab = None
            return

        # Scrollable text for weight display
        text_w = tk.Text(outer, font=MONO_FONT, wrap=tk.WORD,
                         bg="#1e1e2e", fg="#cdd6f4", insertbackground="#cdd6f4",
                         selectbackground="#45475a", relief=tk.FLAT, padx=8, pady=6)
        text_w.pack(fill=tk.BOTH, expand=True)

        if isinstance(weights, list):
            # Neural network: list of weight matrices
            for i, w in enumerate(weights, 1):
                text_w.insert(tk.END, f"W^{i} (shape {w.shape}):\n{np.round(w, 8)}\n\n")
        else:
            w_arr = np.atleast_2d(weights)
            poly_labels = result.get('poly_labels')
            if poly_labels and len(poly_labels) == w_arr.shape[0]:
                text_w.insert(tk.END, f"W (shape {w_arr.shape}):\n")
                w_rounded = np.round(w_arr, 8)
                max_label_len = max(len(l) for l in poly_labels)
                for i, label in enumerate(poly_labels):
                    text_w.insert(tk.END, f"  {label:<{max_label_len}} : {w_rounded[i].tolist()}\n")
                text_w.insert(tk.END, "\n")
            else:
                text_w.insert(tk.END, f"W (shape {w_arr.shape}):\n{np.round(w_arr, 8)}\n\n")

        if bias is not None:
            b_arr = np.atleast_1d(bias)
            text_w.insert(tk.END, f"b (intercept): {np.round(b_arr, 8)}\n")

        text_w.configure(state=tk.DISABLED)

        self._weights_tab = outer
        self.output_notebook.add(outer, text="Weights")

    def _build_tree_tab(self, tree_node):
        """Build or rebuild the Tree visualization tab."""
        self._remove_tab(self._tree_tab)
        if tree_node is None:
            self._tree_tab = None
            return

        viz = TreeVisualizer(self.output_notebook, tree_node)
        self._tree_tab = viz
        self.output_notebook.add(viz, text="Tree")

    def show_result(self, result_dict):
        """Populate tabs from a structured result dict."""
        self._last_result = result_dict

        # Summary tab
        summary_text = result_dict.get('summary_text', result_dict.get('text', ''))
        self.summary_tab.configure(state=tk.NORMAL)
        self.summary_tab.delete("1.0", tk.END)
        self.summary_tab.insert(tk.END, summary_text)
        self.summary_tab.configure(state=tk.DISABLED)

        # Iterations tab
        iterations = result_dict.get('iterations', [])
        self._remove_tab(self._iterations_tab)
        self._iterations_tab = None
        if iterations:
            self._build_iterations_tab(iterations)

        # Weights tab
        self._remove_tab(self._weights_tab)
        self._weights_tab = None
        if result_dict.get('weights') is not None:
            self._build_weights_tab(result_dict)

        # Tree tab
        self._remove_tab(self._tree_tab)
        self._tree_tab = None
        if result_dict.get('tree') is not None:
            self._build_tree_tab(result_dict['tree'])

        # Select Summary tab
        self.output_notebook.select(self.summary_tab)

    def show_output(self, text):
        """Backward-compatible: accept string or dict."""
        if isinstance(text, dict):
            self.show_result(text)
            return
        self._last_result = None
        # Remove dynamic tabs
        self._remove_tab(self._iterations_tab)
        self._iterations_tab = None
        self._remove_tab(self._weights_tab)
        self._weights_tab = None
        self._remove_tab(self._tree_tab)
        self._tree_tab = None

        self.summary_tab.configure(state=tk.NORMAL)
        self.summary_tab.delete("1.0", tk.END)
        self.summary_tab.insert(tk.END, text)
        self.summary_tab.configure(state=tk.DISABLED)
        self.summary_tab.see(tk.END)

    def clear_output(self):
        self._last_result = None
        self.summary_tab.configure(state=tk.NORMAL)
        self.summary_tab.delete("1.0", tk.END)
        self.summary_tab.configure(state=tk.DISABLED)
        self._remove_tab(self._iterations_tab)
        self._iterations_tab = None
        self._remove_tab(self._weights_tab)
        self._weights_tab = None
        self._remove_tab(self._tree_tab)
        self._tree_tab = None
        self._reset_layout()

    def copy_output(self):
        """Copy from the currently active tab."""
        self.clipboard_clear()
        try:
            current = self.output_notebook.nametowidget(self.output_notebook.select())
        except (tk.TclError, KeyError):
            current = self.summary_tab

        if isinstance(current, tk.Text):
            self.clipboard_append(current.get("1.0", tk.END))
        elif hasattr(current, 'winfo_children'):
            # Try to find a Text widget inside
            for child in current.winfo_children():
                if isinstance(child, tk.Text):
                    self.clipboard_append(child.get("1.0", tk.END))
                    return
            # Fallback: copy full text from last result
            if self._last_result:
                self.clipboard_append(self._last_result.get('text', ''))
            else:
                self.clipboard_append(self.summary_tab.get("1.0", tk.END))

    def run(self):
        raise NotImplementedError

    def _toggle(self, widget, show, **pack_kwargs):
        if show:
            widget.pack(**pack_kwargs)
        else:
            widget.pack_forget()

    # --- Widget helpers ---
    def add_text_input(self, parent, label, hint="", height=3, width=50):
        fr = tk.Frame(parent, bg=MAIN_BG)
        fr.pack(fill=tk.X, pady=2)
        ttk.Label(fr, text=label).pack(anchor="w")
        txt = tk.Text(fr, height=height, width=width, font=MONO_FONT, relief=tk.SOLID, bd=1)
        txt.pack(fill=tk.X)
        if hint:
            ttk.Label(fr, text=hint, foreground="#888", font=(FONT_FAMILY, 8)).pack(anchor="w")
        return txt

    def add_entry(self, parent, label, default="", width=12):
        fr = tk.Frame(parent, bg=MAIN_BG)
        fr.pack(side=tk.LEFT, padx=(0, 10), pady=2)
        ttk.Label(fr, text=label).pack(anchor="w")
        var = tk.StringVar(value=default)
        ent = ttk.Entry(fr, textvariable=var, width=width)
        ent.pack()
        var._frame = fr
        return var

    def add_combo(self, parent, label, values, default=None, width=12):
        fr = tk.Frame(parent, bg=MAIN_BG)
        fr.pack(side=tk.LEFT, padx=(0, 10), pady=2)
        ttk.Label(fr, text=label).pack(anchor="w")
        var = tk.StringVar(value=default or values[0])
        cb = ttk.Combobox(fr, textvariable=var, values=values, width=width, state="readonly")
        cb.pack()
        var._frame = fr
        var._combobox = cb
        return var

    def add_button_group(self, parent, label, values, default=None, on_change=None):
        """A row of toggle buttons for selecting one value, like radio buttons."""
        fr = tk.Frame(parent, bg=MAIN_BG)
        fr.pack(side=tk.LEFT, padx=(0, 10), pady=2)
        ttk.Label(fr, text=label).pack(anchor="w")
        btn_row = tk.Frame(fr, bg=MAIN_BG)
        btn_row.pack(anchor="w")
        var = tk.StringVar(value=default or values[0])
        btns = {}

        def _select(val):
            var.set(val)
            _update()
            if on_change:
                on_change()

        def _update():
            cur = var.get()
            for v, b in btns.items():
                if v == cur:
                    b._bg_color = ACCENT
                    b._fg_color = "#fff"
                    b._hover_color = "#3a3d5c"
                    b._draw(ACCENT)
                else:
                    b._bg_color = "#e0e0e0"
                    b._fg_color = "#222"
                    b._hover_color = "#c8c8c8"
                    b._draw("#e0e0e0")

        for i, val in enumerate(values):
            b = RoundedButton(btn_row, text=val, command=lambda v=val: _select(v),
                              font=(FONT_FAMILY, 10), padx=10, pady=2,
                              bg_color="#e0e0e0", fg_color="#222",
                              hover_color="#c8c8c8", press_color="#b0b0b0")
            b.pack(side=tk.LEFT, padx=(0, 3))
            btns[val] = b

        _update()
        var._frame = fr
        var._btns = btns
        var._update = _update
        return var

    def add_check(self, parent, label, default=False):
        fr = tk.Frame(parent, bg=MAIN_BG)
        fr.pack(side=tk.LEFT, padx=(0, 10), pady=2)
        var = tk.BooleanVar(value=default)
        ToggleSwitch(fr, variable=var, text=label).pack()
        var._frame = fr
        return var

    def get_text(self, widget):
        return widget.get("1.0", tk.END).strip()

    def add_matrix_grid(self, parent, label, rows=2, cols=2, vector_mode=False,
                        row_label="samples", col_label="features", on_resize=None,
                        hide_rows=False, hide_cols=False):
        grid = MatrixGrid(parent, label, rows, cols, vector_mode, row_label, col_label,
                          on_resize=on_resize, hide_rows=hide_rows, hide_cols=hide_cols)
        grid.pack(fill=tk.X, pady=2)
        return grid
