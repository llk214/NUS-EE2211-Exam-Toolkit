import tkinter as tk
from tkinter import ttk

from gui.constants import MAIN_BG, ACCENT, FONT_FAMILY
from gui.widgets import RoundedButton
from gui.base_frame import ModuleFrame
from gui.compute.linear_programming import compute_lp


class LinearProgramFrame(ModuleFrame):
    def __init__(self, parent):
        super().__init__(parent, "Linear Programming")
        f = self.input_frame

        # Objective
        row_obj = tk.Frame(f, bg=MAIN_BG)
        row_obj.pack(fill=tk.X, pady=2)
        ttk.Label(row_obj, text="Objective coefficients").pack(anchor="w")
        self.obj_txt = tk.Text(row_obj, height=1, width=40,
                               font=("Consolas", 11), relief=tk.SOLID, bd=1)
        self.obj_txt.pack(fill=tk.X)
        self.obj_txt.insert("1.0", "7 2 6")
        ttk.Label(row_obj, text="Space-separated coefficients, e.g. \"7 2 6\" for 7x\u2081 + 2x\u2082 + 6x\u2083",
                  foreground="#888", font=(FONT_FAMILY, 8)).pack(anchor="w")

        # Max/Min + toggles
        row_ctrl = tk.Frame(f, bg=MAIN_BG)
        row_ctrl.pack(fill=tk.X, pady=4)
        self.opt_var = self.add_button_group(row_ctrl, "Optimize", ["Maximize", "Minimize"], "Maximize")
        self.nonneg_var = self.add_check(row_ctrl, "Non-negativity (x\u2265 0)", default=True)
        self.integer_var = self.add_check(row_ctrl, "Integer variables (ILP)", default=False)

        # Constraints
        self.constr_txt = self.add_text_input(
            f, "Constraints (one per line)",
            hint="Format: coefficients <= or >= bound, e.g. \"2 1 3 <= 14\"",
            height=6)
        self.constr_txt.insert("1.0",
                               "2 1 3 <= 14\n-1 4 1 >= 3\n3 -2 2 <= 12\n1 5 2 <= 20\n1 1 3 >= 5")

    def run(self):
        try:
            result = compute_lp(
                objective_str=self.get_text(self.obj_txt),
                maximize=(self.opt_var.get() == "Maximize"),
                constraints_str=self.get_text(self.constr_txt),
                non_negative=self.nonneg_var.get(),
                integer=self.integer_var.get(),
            )
            self.show_result(result)
        except Exception as e:
            self.show_output(f"ERROR: {e}")
