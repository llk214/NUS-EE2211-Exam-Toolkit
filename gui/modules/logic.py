import tkinter as tk
from tkinter import ttk

from gui.constants import MAIN_BG, ACCENT, FONT_FAMILY, MONO_FONT
from gui.widgets import RoundedButton
from gui.base_frame import ModuleFrame
from gui.compute.logic import compute_logic


class LogicFrame(ModuleFrame):
    OPERATORS = [
        ("\u00ac", "\u00ac"),   # NOT
        ("\u2227", "\u2227"),   # AND
        ("\u2228", "\u2228"),   # OR
        ("\u21d2", "\u21d2"),   # IMPLIES
        ("\u21d4", "\u21d4"),   # IFF
        ("\u2261", "\u21d4"),   # EQUIVALENT (inserts ⇔)
        ("(", "("),
        (")", ")"),
    ]
    PROPS = list("ABCDEFGH")

    def __init__(self, parent):
        super().__init__(parent, "Propositional Logic")
        f = self.input_frame

        # Track the last focused text widget for button insertion
        self._active_txt = None

        # Mode selector
        row_mode = tk.Frame(f, bg=MAIN_BG)
        row_mode.pack(fill=tk.X, pady=4)
        self.mode_var = self.add_button_group(
            row_mode, "Mode",
            ["Truth Table", "Entailment", "Check", "Equivalence"],
            "Truth Table", on_change=self._on_mode_change)

        # Expression input (used by Truth Table, Check, Equivalence — hidden in Entailment)
        self._expr_frame = tk.Frame(f, bg=MAIN_BG)
        self.expr_txt = self._make_text(self._expr_frame, "Expression",
            hint="Use buttons below or type: ~ & | -> <->  (propositions: A-H)",
            height=2)

        # Operator buttons — insert into whichever text widget was last focused
        op_row = tk.Frame(f, bg=MAIN_BG)
        op_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(op_row, text="Operators:", font=(FONT_FAMILY, 9)).pack(side=tk.LEFT, padx=(0, 6))
        for label, sym in self.OPERATORS:
            RoundedButton(op_row, text=label,
                          command=lambda s=sym: self._insert_active(s),
                          font=(FONT_FAMILY, 11), padx=8, pady=2,
                          bg_color="#e0e0e0", fg_color="#222",
                          hover_color="#c8c8c8", press_color="#b0b0b0"
                          ).pack(side=tk.LEFT, padx=(0, 3))

        # Proposition buttons
        prop_row = tk.Frame(f, bg=MAIN_BG)
        prop_row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(prop_row, text="Propositions:", font=(FONT_FAMILY, 9)).pack(side=tk.LEFT, padx=(0, 6))
        for p in self.PROPS:
            RoundedButton(prop_row, text=p,
                          command=lambda s=p: self._insert_active(s),
                          font=(FONT_FAMILY, 10, "bold"), padx=8, pady=2,
                          bg_color=ACCENT, fg_color="#fff",
                          hover_color="#3a3d5c", press_color="#1a1d32"
                          ).pack(side=tk.LEFT, padx=(0, 3))

        # --- Mode-dependent inputs ---

        # Entailment: KB + queries
        self._kb_frame = tk.Frame(f, bg=MAIN_BG)
        self.kb_txt = self._make_text(self._kb_frame, "KB Statements (one per line)", height=4)
        self.query_txt = self._make_text(self._kb_frame, "Queries (one per line)", height=3)

        # Equivalence: second expression
        self._expr2_frame = tk.Frame(f, bg=MAIN_BG)
        self.expr2_txt = self._make_text(self._expr2_frame, "Expression 2",
            hint="Same syntax as Expression above", height=2)

        self._on_mode_change()

    def _make_text(self, parent, label, hint="", height=3):
        """Create a labeled text widget with Copy/Paste/Clear/Undo buttons."""
        # Header row: label + buttons
        header = tk.Frame(parent, bg=MAIN_BG)
        header.pack(fill=tk.X)
        ttk.Label(header, text=label).pack(side=tk.LEFT)

        # Undo snapshot for this text widget
        undo_state = {'snapshot': None, 'clear_btn': None}

        def _copy():
            content = txt.get("1.0", tk.END).rstrip('\n')
            self.winfo_toplevel().clipboard_clear()
            self.winfo_toplevel().clipboard_append(content)

        def _paste():
            try:
                content = self.winfo_toplevel().clipboard_get()
            except tk.TclError:
                return
            txt.insert(tk.INSERT, content)

        def _clear_or_undo():
            btn = undo_state['clear_btn']
            if undo_state['snapshot'] is not None:
                # Undo: restore
                txt.delete("1.0", tk.END)
                txt.insert("1.0", undo_state['snapshot'])
                undo_state['snapshot'] = None
                _set_btn_mode(btn, "clear")
            else:
                # Clear: save snapshot first
                content = txt.get("1.0", tk.END).rstrip('\n')
                if not content:
                    return
                undo_state['snapshot'] = content
                txt.delete("1.0", tk.END)
                _set_btn_mode(btn, "undo")
                # If user types, discard undo
                txt.bind("<Key>", lambda e: _discard_undo(), add="+")

        def _discard_undo():
            if undo_state['snapshot'] is not None:
                undo_state['snapshot'] = None
                _set_btn_mode(undo_state['clear_btn'], "clear")
                txt.unbind("<Key>")

        def _set_btn_mode(btn, mode):
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

        RoundedButton(header, text="Paste", width=5, command=_paste,
                      font=(FONT_FAMILY, 9), pady=1).pack(side=tk.LEFT, padx=(6, 0))
        RoundedButton(header, text="Copy", width=5, command=_copy,
                      font=(FONT_FAMILY, 9), pady=1).pack(side=tk.LEFT, padx=(2, 0))
        clear_btn = RoundedButton(header, text="Clear", width=5, command=_clear_or_undo,
                      font=(FONT_FAMILY, 9), pady=1)
        clear_btn.pack(side=tk.LEFT, padx=(2, 0))
        undo_state['clear_btn'] = clear_btn

        txt = tk.Text(parent, height=height, width=50, font=MONO_FONT, relief=tk.SOLID, bd=1)
        txt.pack(fill=tk.X, pady=(0, 2))
        if hint:
            ttk.Label(parent, text=hint, foreground="#888", font=(FONT_FAMILY, 8)).pack(anchor="w")
        # Track focus so operator/proposition buttons insert into the right widget
        txt.bind("<FocusIn>", lambda e, w=txt: self._set_active(w))
        return txt

    def _set_active(self, widget):
        self._active_txt = widget

    def _insert_active(self, text):
        """Insert text into the last focused text widget."""
        target = self._active_txt
        if target is None:
            # Default based on mode
            mode = self.mode_var.get()
            if mode == "Entailment":
                target = self.kb_txt
            else:
                target = self.expr_txt
        target.insert(tk.INSERT, text)
        target.focus_set()

    def _on_mode_change(self):
        mode = self.mode_var.get()
        # Expression box: shown for Truth Table, Check, Equivalence — hidden for Entailment
        self._toggle(self._expr_frame, mode != "Entailment", fill=tk.X, pady=2)
        self._toggle(self._kb_frame, mode == "Entailment", fill=tk.X, pady=4)
        self._toggle(self._expr2_frame, mode == "Equivalence", fill=tk.X, pady=4)
        # Reset active target when switching modes
        self._active_txt = None

    def run(self):
        try:
            mode = self.mode_var.get()
            expr = self.get_text(self.expr_txt)

            if mode == "Truth Table":
                result = compute_logic('truth_table', expression=expr)
            elif mode == "Entailment":
                kb = self.get_text(self.kb_txt).splitlines()
                queries = self.get_text(self.query_txt).splitlines()
                result = compute_logic('entailment', kb_lines=kb, query_lines=queries)
            elif mode == "Check":
                result = compute_logic('check', expression=expr)
            elif mode == "Equivalence":
                expr2 = self.get_text(self.expr2_txt)
                result = compute_logic('equivalence', expression=expr, expression2=expr2)
            else:
                result = {'text': 'Unknown mode', 'summary_text': 'Unknown mode'}

            self.show_result(result)
        except Exception as e:
            self.show_output(f"ERROR: {e}")
