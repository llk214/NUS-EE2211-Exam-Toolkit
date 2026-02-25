import tkinter as tk
from tkinter import ttk
import re

from gui.constants import MAIN_BG, FONT_FAMILY, MONO_FONT
from gui.widgets import RoundedButton

try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sympy import symbols as _sp_symbols, latex as _sp_latex
    from sympy.parsing.sympy_parser import (
        parse_expr as _sp_parse,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    _SP_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
    _SP_VARS = {}
    for _n in ('x1', 'x2', 'x3', 'x4', 'x5'):
        _SP_VARS[_n] = _sp_symbols(_n)
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


def expr_to_latex(expr):
    """Convert a raw expression string (with ** for powers) to LaTeX using sympy."""
    s = expr.strip()
    if not s or not HAS_SYMPY:
        return s
    try:
        parsed = _sp_parse(s, local_dict=_SP_VARS, transformations=_SP_TRANSFORMS)
        return _sp_latex(parsed)
    except Exception:
        return s


_SUBSCRIPTS = str.maketrans('12345', '\u2081\u2082\u2083\u2084\u2085')


class ExpressionEditor(tk.Frame):
    """Single-line expression editor where variables are embedded chip widgets,
    insertable only via button press."""

    def __init__(self, parent, font=None, **kwargs):
        super().__init__(parent, bg=MAIN_BG)
        self._font = font or MONO_FONT
        self._text = tk.Text(self, height=1, wrap=tk.NONE, font=self._font,
                             relief=tk.SOLID, bd=1, undo=True)
        self._text.pack(fill=tk.X)
        self._text.bind("<Return>", lambda e: "break")
        self._text.bind("<KeyRelease>", lambda e: self._fire_change())
        self._text.bind("<<Paste>>", lambda e: self.after(10, self._fire_change))
        self._chip_map = {}  # str(widget) -> var_name
        self._on_change_cb = None

    # -- public API used by MathKeyboard (Entry-compatible shims) --

    def insert(self, index, text):
        self._text.insert(index, text)
        self._fire_change()

    def index(self, idx):
        """Return integer column (single-line assumption)."""
        s = self._text.index(idx)
        return int(s.split('.')[1])

    def icursor(self, pos):
        self._text.mark_set(tk.INSERT, f"1.{pos}")

    def focus_set(self):
        self._text.focus_set()

    # -- chip insertion --

    def insert_variable(self, var_name):
        """Insert a variable chip at the cursor position."""
        display = 'x' + var_name[1:].translate(_SUBSCRIPTS)
        chip = tk.Label(self._text, text=display,
                        bg="#dce6f7", fg="#1a56db",
                        font=(FONT_FAMILY, 9),
                        padx=2, pady=0, bd=0)
        self._text.window_create(tk.INSERT, window=chip, padx=1)
        self._chip_map[str(chip)] = var_name
        self._text.focus_set()
        self._fire_change()

    # -- expression extraction --

    def get_expression(self):
        """Return the expression string with chip positions replaced by var names.
        Inserts '*' for implicit multiplication (e.g. x1 x2, 6sin(), 5x1)."""
        parts = []
        prev_was_chip = False
        try:
            for key, val, idx in self._text.dump("1.0", "end-1c",
                                                  text=True, window=True):
                if key == 'text':
                    # Insert * between chip and text that starts with digit/letter/(
                    if prev_was_chip and val and val[0] in '0123456789.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ(':
                        parts.append('*')
                    parts.append(val)
                    prev_was_chip = False
                elif key == 'window':
                    name = self._chip_map.get(val)
                    if name:
                        # Insert * if previous part ends with digit/letter/)
                        if prev_was_chip:
                            parts.append('*')
                        elif parts:
                            prev_text = parts[-1]
                            if prev_text and prev_text[-1] in '0123456789.)abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                                parts.append('*')
                        parts.append(name)
                        prev_was_chip = True
        except tk.TclError:
            pass
        raw = ''.join(parts).replace('^', '**')
        # Insert * between: digit and letter/( , ) and letter/digit/(
        raw = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', raw)
        raw = re.sub(r'(\))([a-zA-Z0-9(])', r'\1*\2', raw)
        return raw

    def get_raw_text(self):
        """Return only the typed text (excluding chip widgets)."""
        parts = []
        try:
            for key, val, idx in self._text.dump("1.0", "end-1c", text=True):
                if key == 'text':
                    parts.append(val)
        except tk.TclError:
            pass
        return ''.join(parts)

    def get(self, *args):
        """Compat shim: return full expression (^ converted to **)."""
        return self.get_expression()

    def get_variables(self):
        """Return sorted list of unique variable names currently present as chips."""
        found = set()
        try:
            for key, val, idx in self._text.dump("1.0", "end-1c", window=True):
                if key == 'window':
                    name = self._chip_map.get(val)
                    if name:
                        found.add(name)
        except tk.TclError:
            pass
        return sorted(found)

    # -- change callback --

    def set_on_change(self, callback):
        self._on_change_cb = callback

    def _fire_change(self):
        if self._on_change_cb:
            self._on_change_cb()


class MathPreview(tk.Frame):
    """Displays a live-rendered LaTeX preview of the cost function expression."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=MAIN_BG)
        self._debounce_id = None

        self._fig = Figure(figsize=(6, 0.5), dpi=100)
        self._fig.patch.set_facecolor(MAIN_BG)
        self._ax = self._fig.add_axes([0, 0, 1, 1])
        self._ax.set_axis_off()

        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill=tk.X)

        self._text_obj = None

    def update_expression(self, raw_expr):
        """Schedule a debounced preview update."""
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(150, lambda: self._render(raw_expr))

    def _render(self, raw_expr):
        """Render the expression onto the matplotlib figure."""
        self._debounce_id = None
        self._ax.clear()
        self._ax.set_axis_off()

        expr = raw_expr.strip()
        if not expr:
            self._canvas.draw_idle()
            return

        latex = expr_to_latex(expr)
        display = r'$C = ' + latex + r'$'

        try:
            self._ax.text(0.02, 0.5, display, fontsize=14,
                          verticalalignment='center',
                          transform=self._ax.transAxes,
                          color='#89b4fa', usetex=False)
            self._canvas.draw()
        except Exception:
            # Fallback: show raw expression with ^ instead of **
            self._ax.clear()
            self._ax.set_axis_off()
            fallback = 'C = ' + expr.replace('**', '^')
            self._ax.text(0.02, 0.5, fallback, fontsize=12,
                          verticalalignment='center',
                          transform=self._ax.transAxes,
                          color='#888888', usetex=False)
            self._canvas.draw_idle()


class MathKeyboard(tk.Frame):
    """Clickable math function button pad for the Cost Minimizer."""

    BUTTONS = [
        # (label, insert_text, cursor_back)
        ("sin", "sin()", 1), ("cos", "cos()", 1), ("tan", "tan()", 1),
        ("exp", "exp()", 1), ("ln", "ln()", 1), ("sqrt", "sqrt()", 1),
        ("\u03c0", "pi", 0), ("e", "e", 0),
        ("^", "^", 0), ("(", "(", 0), (")", ")", 0),
        ("*", "*", 0), ("/", "/", 0), ("+", "+", 0),
        ("-", "-", 0), ("^2", "^2", 0),
    ]

    def __init__(self, parent, target_widget):
        super().__init__(parent, bg=MAIN_BG)
        self.target = target_widget
        for i, (label, text, back) in enumerate(self.BUTTONS):
            btn = RoundedButton(self, text=label, width=5,
                                command=lambda t=text, b=back: self._insert(t, b),
                                font=(FONT_FAMILY, 9), pady=1)
            btn.grid(row=i // 8, column=i % 8, padx=1, pady=1)

    def _insert(self, text, cursor_back):
        self.target.insert(tk.INSERT, text)
        if cursor_back > 0:
            pos = self.target.index(tk.INSERT)
            self.target.icursor(pos - cursor_back)
        self.target.focus_set()
