import tkinter as tk
from tkinter import ttk
import numpy as np

from gui.constants import MAIN_BG, ACCENT, FONT_FAMILY, MONO_FONT, GRID_CELL_FONT
from gui.compute.tree import TreeNode


class ToggleSwitch(tk.Canvas):
    """A pill-shaped toggle switch drawn on a Canvas."""

    def __init__(self, parent, variable=None, text="", font=None,
                 width=40, height=22, on_color=None, off_color="#ccc",
                 knob_color="#fff", text_color="#222", **kwargs):
        bg = parent.cget("bg") if hasattr(parent, "cget") else MAIN_BG
        super().__init__(parent, highlightthickness=0, bd=0, bg=bg, **kwargs)
        self._var = variable or tk.BooleanVar(value=False)
        self._sw = width
        self._sh = height
        self._on_color = on_color or ACCENT
        self._off_color = off_color
        self._knob_color = knob_color
        self._font = font or (FONT_FAMILY, 11)
        self._text = text

        # Measure text for total widget width
        if text:
            _tmp = tk.Label(self, text=text, font=self._font)
            _tmp.update_idletasks()
            self._tw = _tmp.winfo_reqwidth()
            self._th = _tmp.winfo_reqheight()
            _tmp.destroy()
            self._gap = 6
        else:
            self._tw = 0
            self._th = 0
            self._gap = 0

        total_w = self._sw + self._gap + self._tw
        total_h = max(self._sh, self._th)
        self.configure(width=total_w, height=total_h)

        self._draw()
        self.bind("<ButtonRelease-1>", self._on_click)
        self._var.trace_add("write", lambda *a: self._draw())

    def _draw(self):
        self.delete("all")
        on = self._var.get()
        sw, sh = self._sw, self._sh
        total_h = int(self.cget("height"))
        y_off = (total_h - sh) // 2
        r = sh // 2
        pad = 3
        knob_r = r - pad

        # Track color
        color = self._on_color if on else self._off_color

        # Draw pill track
        self.create_arc(0, y_off, sh, y_off + sh, start=90, extent=180, fill=color, outline=color)
        self.create_arc(sw - sh, y_off, sw, y_off + sh, start=270, extent=180, fill=color, outline=color)
        self.create_rectangle(r, y_off, sw - r, y_off + sh, fill=color, outline=color)

        # Draw knob
        if on:
            cx = sw - r
        else:
            cx = r
        cy = y_off + r
        self.create_oval(cx - knob_r, cy - knob_r, cx + knob_r, cy + knob_r,
                         fill=self._knob_color, outline=self._knob_color)

        # Draw label text
        if self._text:
            tx = sw + self._gap
            ty = total_h // 2
            self.create_text(tx, ty, text=self._text, font=self._font,
                             fill="#222", anchor="w")

    def _on_click(self, event):
        self._var.set(not self._var.get())


class RoundedButton(tk.Canvas):
    """A button with rounded corners drawn on a Canvas."""

    def __init__(self, parent, text="", command=None, width=None, radius=6,
                 font=None, bg_color="#e0e0e0", fg_color="#222", hover_color="#c8c8c8",
                 press_color="#b0b0b0", padx=6, pady=2, **kwargs):
        super().__init__(parent, highlightthickness=0, bd=0,
                         bg=parent.cget("bg") if hasattr(parent, "cget") else MAIN_BG,
                         **kwargs)
        self._text = text
        self._command = command
        self._radius = radius
        self._bg_color = bg_color
        self._fg_color = fg_color
        self._hover_color = hover_color
        self._press_color = press_color
        self._padx = padx
        self._pady = pady
        self._font = font or (FONT_FAMILY, 10)
        self._disabled = False

        # Measure text to determine canvas size
        _tmp = tk.Label(self, text=text, font=self._font)
        _tmp.update_idletasks()
        tw = _tmp.winfo_reqwidth()
        th = _tmp.winfo_reqheight()
        _tmp.destroy()

        if width is not None:
            # width in approximate character widths
            cw = tk.Label(self, text="0", font=self._font)
            cw.update_idletasks()
            char_w = cw.winfo_reqwidth()
            cw.destroy()
            tw = max(tw, int(width * char_w * 0.75))

        self._btn_w = tw + padx * 2
        self._btn_h = th + pady * 2
        self.configure(width=self._btn_w, height=self._btn_h)

        self._draw(self._bg_color)

        self.bind("<Enter>", lambda e: self._draw(self._hover_color) if not self._disabled else None)
        self.bind("<Leave>", lambda e: self._draw(self._bg_color) if not self._disabled else None)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw(self, fill):
        self.delete("all")
        r = self._radius
        w, h = self._btn_w, self._btn_h
        # Rounded rectangle via arcs + rectangles
        self.create_arc(0, 0, r * 2, r * 2, start=90, extent=90, fill=fill, outline=fill)
        self.create_arc(w - r * 2, 0, w, r * 2, start=0, extent=90, fill=fill, outline=fill)
        self.create_arc(0, h - r * 2, r * 2, h, start=180, extent=90, fill=fill, outline=fill)
        self.create_arc(w - r * 2, h - r * 2, w, h, start=270, extent=90, fill=fill, outline=fill)
        self.create_rectangle(r, 0, w - r, h, fill=fill, outline=fill)
        self.create_rectangle(0, r, w, h - r, fill=fill, outline=fill)
        self.create_text(w // 2, h // 2, text=self._text, font=self._font, fill=self._fg_color)

    def _on_press(self, event):
        if not self._disabled:
            self._draw(self._press_color)

    def _on_release(self, event):
        if not self._disabled:
            self._draw(self._hover_color)
            if self._command and 0 <= event.x <= self._btn_w and 0 <= event.y <= self._btn_h:
                self._command()

    def configure_state(self, state):
        self._disabled = (state == "disabled")


class MatrixGrid(tk.Frame):
    """A graphing-calculator-style grid widget for matrix/vector input."""

    def __init__(self, parent, label, rows=2, cols=2, vector_mode=False,
                 row_label="samples", col_label="features", on_resize=None,
                 hide_rows=False, hide_cols=False):
        super().__init__(parent, bg=MAIN_BG)
        self.label = label
        self.n_rows = rows
        self.n_cols = cols
        self.vector_mode = vector_mode
        self.row_label = row_label
        self.col_label = col_label
        self.cells = []  # 2D list of Entry widgets
        self.on_resize = on_resize  # callback(new_rows, new_cols)

        # Header row: label + dimension controls
        header = tk.Frame(self, bg=MAIN_BG)
        header.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(header, text=label, font=(FONT_FAMILY, 10, "bold")).pack(side=tk.LEFT)

        self.rows_var = tk.StringVar(value=str(rows))
        if not vector_mode:
            if not hide_rows:
                ttk.Label(header, text="Rows:").pack(side=tk.LEFT, padx=(8, 0))
                rows_spin = ttk.Spinbox(header, from_=1, to=50, textvariable=self.rows_var,
                                        width=3, command=self._on_spin_change)
                rows_spin.pack(side=tk.LEFT, padx=(2, 4))

            if not hide_cols:
                ttk.Label(header, text="Cols:").pack(side=tk.LEFT, padx=(8, 0) if hide_rows else (0, 0))
            self.cols_var = tk.StringVar(value=str(cols))
            if not hide_cols:
                cols_spin = ttk.Spinbox(header, from_=1, to=50, textvariable=self.cols_var,
                                        width=3, command=self._on_spin_change)
                cols_spin.pack(side=tk.LEFT, padx=(2, 4))
        else:
            ttk.Label(header, text="Size:").pack(side=tk.LEFT, padx=(8, 0))
            rows_spin = ttk.Spinbox(header, from_=1, to=50, textvariable=self.rows_var,
                                    width=3, command=self._on_spin_change)
            rows_spin.pack(side=tk.LEFT, padx=(2, 4))
            self.cols_var = tk.StringVar(value="1")

        RoundedButton(header, text="Paste", width=5, command=self._paste_from_clipboard,
                      font=(FONT_FAMILY, 9), pady=1).pack(side=tk.LEFT, padx=(6, 0))
        RoundedButton(header, text="Copy", width=5, command=self._copy_to_clipboard,
                      font=(FONT_FAMILY, 9), pady=1).pack(side=tk.LEFT, padx=(2, 0))
        self._clear_btn = RoundedButton(header, text="Clear", width=5, command=self._clear_or_undo,
                      font=(FONT_FAMILY, 9), pady=1)
        self._clear_btn.pack(side=tk.LEFT, padx=(2, 0))
        self._undo_snapshot = None  # (rows, cols, 2D list of strings)

        # Grid area
        self.grid_outer = tk.Frame(self, bg="#e8e8e8", bd=1, relief=tk.SOLID)
        self.grid_outer.pack(anchor="w", pady=2)

        self.grid_inner = tk.Frame(self.grid_outer, bg="#f0f0f0")
        self.grid_inner.pack()

        # Shape label
        self.shape_label = ttk.Label(self, text="", foreground="#888", font=(FONT_FAMILY, 8))
        self.shape_label.pack(anchor="w")

        self._build_grid()

    @staticmethod
    def _validate_cell(new_value):
        """Allow only digits, dots, minus signs, and empty string."""
        if new_value == "":
            return True
        for ch in new_value:
            if ch not in "0123456789.-":
                return False
        return True

    def _on_cell_focus_out(self, event):
        ent = event.widget
        if not ent.get().strip():
            ent.insert(0, "0")

    def _build_grid(self):
        for widget in self.grid_inner.winfo_children():
            widget.destroy()
        self.cells = []

        vcmd = (self.register(self._validate_cell), '%P')

        for r in range(self.n_rows):
            row_cells = []
            # Left bracket
            tk.Label(self.grid_inner, text="[", font=("Consolas", 14, "bold"),
                     bg="#f0f0f0", fg="#555").grid(row=r, column=0, padx=(4, 0))
            for c in range(self.n_cols):
                ent = tk.Entry(self.grid_inner, width=8, font=GRID_CELL_FONT,
                               justify=tk.CENTER, relief=tk.SOLID, bd=1,
                               validate="key", validatecommand=vcmd)
                ent.grid(row=r, column=c + 1, padx=1, pady=1)
                ent.insert(0, "0")
                ent.bind("<Control-v>", self._on_paste)
                ent.bind("<<Paste>>", self._on_paste)
                ent.bind("<FocusOut>", self._on_cell_focus_out)
                row_cells.append(ent)
            # Right bracket
            tk.Label(self.grid_inner, text="]", font=("Consolas", 14, "bold"),
                     bg="#f0f0f0", fg="#555").grid(row=r, column=self.n_cols + 1, padx=(0, 4))
            self.cells.append(row_cells)

        self._update_shape_label()

    def _on_spin_change(self):
        try:
            new_rows = int(self.rows_var.get())
            new_cols = int(self.cols_var.get()) if not self.vector_mode else 1
        except ValueError:
            return
        new_rows = max(1, min(50, new_rows))
        new_cols = max(1, min(50, new_cols))
        self._resize(new_rows, new_cols)

    def _resize(self, new_rows, new_cols):
        # Save existing values
        old_vals = []
        for r in range(min(self.n_rows, new_rows)):
            row_vals = []
            for c in range(min(self.n_cols, new_cols)):
                row_vals.append(self.cells[r][c].get())
            old_vals.append(row_vals)

        self.n_rows = new_rows
        self.n_cols = new_cols
        self.rows_var.set(str(new_rows))
        if not self.vector_mode:
            self.cols_var.set(str(new_cols))
        self._build_grid()

        # Restore saved values
        for r in range(len(old_vals)):
            for c in range(len(old_vals[r])):
                self.cells[r][c].delete(0, tk.END)
                self.cells[r][c].insert(0, old_vals[r][c])

        if self.on_resize:
            self.on_resize(new_rows, new_cols)

    def _add_row(self):
        self._resize(self.n_rows + 1, self.n_cols)

    def _remove_row(self):
        if self.n_rows > 1:
            self._resize(self.n_rows - 1, self.n_cols)

    def _add_col(self):
        self._resize(self.n_rows, self.n_cols + 1)

    def _remove_col(self):
        if self.n_cols > 1:
            self._resize(self.n_rows, self.n_cols - 1)

    def _snapshot_cells(self):
        """Capture current cell values and grid dimensions."""
        data = []
        for row in self.cells:
            data.append([ent.get() for ent in row])
        return (self.n_rows, self.n_cols, data)

    def _is_all_zero(self):
        """Check if every cell is '0' or empty."""
        return all(
            (ent.get().strip() or "0") == "0"
            for row in self.cells for ent in row
        )

    def _clear_or_undo(self):
        if self._undo_snapshot is not None:
            # Undo: restore snapshot
            rows, cols, data = self._undo_snapshot
            self._undo_snapshot = None
            self._resize(rows, cols)
            for r in range(min(rows, len(data))):
                for c in range(min(cols, len(data[r]))):
                    self.cells[r][c].delete(0, tk.END)
                    self.cells[r][c].insert(0, data[r][c])
            self._set_clear_btn_mode("clear")
        else:
            # Clear: save snapshot first, then zero out
            if self._is_all_zero():
                return
            self._undo_snapshot = self._snapshot_cells()
            self._clear_all()
            self._set_clear_btn_mode("undo")
            # Listen for any edit — if user types, discard undo
            self._bind_edit_watchers()

    def _set_clear_btn_mode(self, mode):
        btn = self._clear_btn
        if mode == "undo":
            btn._text = "Undo"
            btn._bg_color = "#f9e2af"
            btn._fg_color = "#1e1e2e"
            btn._hover_color = "#f2d48f"
            btn._draw("#f9e2af")
        else:
            btn._text = "Clear"
            btn._bg_color = "#e0e0e0"
            btn._fg_color = "#222"
            btn._hover_color = "#c8c8c8"
            btn._draw("#e0e0e0")

    def _bind_edit_watchers(self):
        """Bind a one-shot key listener on every cell to cancel undo on edit."""
        for row in self.cells:
            for ent in row:
                ent.bind("<Key>", self._on_edit_after_clear, add="+")

    def _unbind_edit_watchers(self):
        for row in self.cells:
            for ent in row:
                ent.unbind("<Key>")

    def _on_edit_after_clear(self, event):
        # Ignore modifier-only keys and Tab/Shift-Tab
        if event.keysym in ("Tab", "ISO_Left_Tab", "Shift_L", "Shift_R",
                            "Control_L", "Control_R", "Alt_L", "Alt_R",
                            "Caps_Lock"):
            return
        self._undo_snapshot = None
        self._set_clear_btn_mode("clear")
        self._unbind_edit_watchers()

    def _clear_all(self):
        for row in self.cells:
            for ent in row:
                ent.delete(0, tk.END)
                ent.insert(0, "0")

    def _parse_paste_data(self, text):
        """Parse clipboard text into a 2D list of strings."""
        text = text.strip()
        if not text:
            return [["0"]]

        # Try comma-separated rows format: "1 2, 3 4, 5 6"
        if ',' in text and '\t' not in text:
            rows = [r.strip() for r in text.split(',') if r.strip()]
            data = []
            for r in rows:
                cols = r.split()
                data.append(cols)
            return data

        # Tab-separated (Excel paste)
        if '\t' in text:
            rows = text.split('\n')
            data = []
            for r in rows:
                r = r.strip()
                if r:
                    data.append(r.split('\t'))
            return data

        # Newline-separated rows, space-separated columns
        if '\n' in text:
            rows = text.split('\n')
            data = []
            for r in rows:
                r = r.strip()
                if r:
                    data.append(r.split())
            return data

        # Single row, space-separated
        return [text.split()]

    def _on_paste(self, event=None):
        try:
            text = self.winfo_toplevel().clipboard_get()
        except tk.TclError:
            return
        data = self._parse_paste_data(text)
        if not data:
            return

        new_rows = len(data)
        new_cols = max(len(r) for r in data)
        if self.vector_mode:
            # For vectors, paste as single column
            if new_cols == 1:
                self._resize(new_rows, 1)
            else:
                # Flatten to column
                flat = []
                for r in data:
                    flat.extend(r)
                data = [[v] for v in flat]
                new_rows = len(flat)
                new_cols = 1
                self._resize(new_rows, 1)
        else:
            self._resize(new_rows, new_cols)

        for r in range(new_rows):
            for c in range(new_cols):
                if r < len(data) and c < len(data[r]):
                    self.cells[r][c].delete(0, tk.END)
                    self.cells[r][c].insert(0, data[r][c])
        return "break"

    def _paste_from_clipboard(self):
        self._on_paste()

    def _copy_to_clipboard(self):
        """Copy grid contents to clipboard as tab-separated rows."""
        lines = []
        for row in self.cells:
            lines.append("\t".join(ent.get().strip() or "0" for ent in row))
        text = "\n".join(lines)
        self.winfo_toplevel().clipboard_clear()
        self.winfo_toplevel().clipboard_append(text)

    def _update_shape_label(self):
        if self.vector_mode:
            self.shape_label.config(text=f"{self.n_rows} {self.row_label}")
        else:
            self.shape_label.config(
                text=f"{self.n_rows} {self.row_label} \u00d7 {self.n_cols} {self.col_label}")

    def get_matrix_string(self):
        """Return comma-separated row format: '1 2, 3 4, 5 6'."""
        rows = []
        for row in self.cells:
            vals = []
            for ent in row:
                v = ent.get().strip()
                if not v:
                    v = "0"
                vals.append(v)
            rows.append(" ".join(vals))
        return ", ".join(rows)

    def get_vector_string(self):
        """Return space-separated values for 1D vectors."""
        vals = []
        for row in self.cells:
            for ent in row:
                v = ent.get().strip()
                if not v:
                    v = "0"
                vals.append(v)
        return " ".join(vals)

    def set_from_string(self, s):
        """Populate grid from comma-separated row string."""
        s = s.strip()
        if not s:
            return
        data = self._parse_paste_data(s)
        new_rows = len(data)
        new_cols = max(len(r) for r in data)
        if self.vector_mode:
            new_cols = 1
        self._resize(new_rows, new_cols)
        for r in range(new_rows):
            for c in range(new_cols):
                if r < len(data) and c < len(data[r]):
                    self.cells[r][c].delete(0, tk.END)
                    self.cells[r][c].insert(0, data[r][c])

    def set_from_matrix(self, arr):
        """Resize grid to match arr shape and fill cells from a numpy array."""
        arr = np.atleast_2d(arr)
        self._resize(arr.shape[0], arr.shape[1])
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                self.cells[r][c].delete(0, tk.END)
                self.cells[r][c].insert(0, str(round(float(arr[r, c]), 8)))

    def get_shape(self):
        return (self.n_rows, self.n_cols)


class CollapsibleSection(tk.Frame):
    """A collapsible section with a clickable header and hideable body."""

    def __init__(self, parent, title, body_text, expanded=False):
        super().__init__(parent, bg="#1e1e2e")
        self._expanded = expanded

        # Header bar
        self._header = tk.Frame(self, bg="#2a2a3e", cursor="hand2")
        self._header.pack(fill=tk.X, pady=(1, 0))

        self._arrow = tk.Label(self._header, text="\u25bc" if expanded else "\u25b6",
                               bg="#2a2a3e", fg="#89b4fa", font=("Consolas", 10))
        self._arrow.pack(side=tk.LEFT, padx=(6, 4))

        self._title_lbl = tk.Label(self._header, text=title, bg="#2a2a3e", fg="#cdd6f4",
                                   font=("Consolas", 10), anchor="w")
        self._title_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Body
        self._body = tk.Text(self, font=MONO_FONT, wrap=tk.WORD, height=min(12, body_text.count('\n') + 2),
                             bg="#1e1e2e", fg="#cdd6f4", insertbackground="#cdd6f4",
                             selectbackground="#45475a", relief=tk.FLAT, padx=8, pady=4)
        self._body.insert("1.0", body_text)
        self._body.configure(state=tk.DISABLED)

        if expanded:
            self._body.pack(fill=tk.X, padx=(16, 0))

        # Bind click
        for w in (self._header, self._arrow, self._title_lbl):
            w.bind("<Button-1>", self._toggle)

    def _toggle(self, event=None):
        self._expanded = not self._expanded
        if self._expanded:
            self._arrow.configure(text="\u25bc")
            self._body.pack(fill=tk.X, padx=(16, 0))
        else:
            self._arrow.configure(text="\u25b6")
            self._body.pack_forget()


class TreeVisualizer(tk.Frame):
    """Draws a decision tree on a scrollable Canvas."""

    NODE_W = 140
    NODE_H = 40
    H_GAP = 20
    V_GAP = 60
    INTERNAL_COLOR = "#89b4fa"
    LEAF_COLOR = "#a6e3a1"
    TEXT_COLOR = "#1e1e2e"
    LINE_COLOR = "#6c7086"

    def __init__(self, parent, tree_root):
        super().__init__(parent, bg="#1e1e2e")
        self._tree = tree_root

        self._canvas = tk.Canvas(self, bg="#1e1e2e", highlightthickness=0)
        h_scroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self._canvas.xview)
        v_scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        self._canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._positions = {}
        self._draw_tree()

    def _count_leaves(self, node):
        if node.leaf():
            return 1
        return self._count_leaves(node.l) + self._count_leaves(node.r)

    def _compute_positions(self, node, depth, left, right):
        if node.leaf():
            x = (left + right) / 2
            y = depth * (self.NODE_H + self.V_GAP) + 30
            self._positions[id(node)] = (x, y)
            return

        n_left = self._count_leaves(node.l)
        n_total = self._count_leaves(node.l) + self._count_leaves(node.r)
        split = left + (right - left) * n_left / n_total

        self._compute_positions(node.l, depth + 1, left, split)
        self._compute_positions(node.r, depth + 1, split, right)

        lx, ly = self._positions[id(node.l)]
        rx, ry = self._positions[id(node.r)]
        x = (lx + rx) / 2
        y = depth * (self.NODE_H + self.V_GAP) + 30
        self._positions[id(node)] = (x, y)

    def _draw_node(self, node):
        x, y = self._positions[id(node)]
        hw, hh = self.NODE_W // 2, self.NODE_H // 2

        if node.leaf():
            color = self.LEAF_COLOR
            pred_val = node.pred
            if isinstance(pred_val, (float, np.floating)):
                text = f"pred={pred_val:.4f}"
            else:
                text = f"pred={pred_val}"
        else:
            color = self.INTERNAL_COLOR
            text = f"X[{node.f}] <= {node.t:.4f}"

        # Rounded rectangle
        r = 8
        self._canvas.create_arc(x - hw, y - hh, x - hw + 2 * r, y - hh + 2 * r,
                                start=90, extent=90, fill=color, outline=color)
        self._canvas.create_arc(x + hw - 2 * r, y - hh, x + hw, y - hh + 2 * r,
                                start=0, extent=90, fill=color, outline=color)
        self._canvas.create_arc(x - hw, y + hh - 2 * r, x - hw + 2 * r, y + hh,
                                start=180, extent=90, fill=color, outline=color)
        self._canvas.create_arc(x + hw - 2 * r, y + hh - 2 * r, x + hw, y + hh,
                                start=270, extent=90, fill=color, outline=color)
        self._canvas.create_rectangle(x - hw + r, y - hh, x + hw - r, y + hh,
                                      fill=color, outline=color)
        self._canvas.create_rectangle(x - hw, y - hh + r, x + hw, y + hh - r,
                                      fill=color, outline=color)
        self._canvas.create_text(x, y, text=text, font=("Consolas", 9),
                                 fill=self.TEXT_COLOR)

        if not node.leaf():
            # Draw edges to children
            for child, label in [(node.l, "Yes"), (node.r, "No")]:
                cx, cy = self._positions[id(child)]
                self._canvas.create_line(x, y + hh, cx, cy - hh,
                                         fill=self.LINE_COLOR, width=2)
                mx, my = (x + cx) / 2, (y + hh + cy - hh) / 2
                self._canvas.create_text(mx, my - 8, text=label,
                                         font=("Consolas", 8), fill="#bac2de")
            self._draw_node(node.l)
            self._draw_node(node.r)

    def _draw_tree(self):
        if self._tree is None:
            return
        n_leaves = self._count_leaves(self._tree)
        total_w = max(400, n_leaves * (self.NODE_W + self.H_GAP))
        self._compute_positions(self._tree, 0, 0, total_w)
        self._draw_node(self._tree)

        # Set scroll region
        self._canvas.configure(scrollregion=self._canvas.bbox("all") or (0, 0, 400, 300))
