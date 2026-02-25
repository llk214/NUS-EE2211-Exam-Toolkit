import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import tkinter as tk
from tkinter import ttk

from gui.constants import (SIDEBAR_BG, SIDEBAR_FG, SIDEBAR_ACTIVE,
                           SIDEBAR_HOVER, MAIN_BG, FONT_FAMILY)
from gui.modules.classification import ClassificationFrame
from gui.modules.clustering import ClusteringFrame
from gui.modules.gradient_descent import GradientDescentFrame
from gui.modules.neural import NeuralNetFrame
from gui.modules.regression import RegressionFrame
from gui.modules.tree import DecisionTreeFrame
from gui.modules.cost_minimizer import CostMinimizerFrame
from gui.modules.logic import LogicFrame
from gui.modules.search import SearchFrame
from gui.modules.linear_programming import LinearProgramFrame


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EE2211 Exam Toolkit")
        self.geometry("1280x950")
        self.minsize(900, 650)
        self.configure(bg=MAIN_BG)

        # DPI scaling
        try:
            dpi = self.winfo_fpixels('1i')
            scale_factor = dpi / 72.0
            self.tk.call('tk', 'scaling', scale_factor)
        except Exception:
            pass

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", background=MAIN_BG, font=(FONT_FAMILY, 11))
        style.configure("Header.TLabel", font=(FONT_FAMILY, 16, "bold"), background=MAIN_BG)
        style.configure("TCombobox", font=(FONT_FAMILY, 11))
        style.configure("TSpinbox", arrowsize=17)

        # Sidebar
        self.sidebar = tk.Frame(self, bg=SIDEBAR_BG, width=240)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="EE2211", bg=SIDEBAR_BG, fg=SIDEBAR_FG,
                 font=(FONT_FAMILY, 16, "bold"), pady=12).pack(fill=tk.X)
        tk.Frame(self.sidebar, bg=SIDEBAR_ACTIVE, height=2).pack(fill=tk.X, padx=10, pady=(0, 8))

        # Main content
        self.main_frame = tk.Frame(self, bg=MAIN_BG)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.modules = [
            ("Classification", ClassificationFrame),
            ("Clustering", ClusteringFrame),
            ("Gradient Desc", GradientDescentFrame),
            ("Neural Net", NeuralNetFrame),
            ("Regression", RegressionFrame),
            ("Decision Tree", DecisionTreeFrame),
            ("Cost Minimizer", CostMinimizerFrame),
            ("Logic", LogicFrame),
            ("Search", SearchFrame),
            ("Linear Prog", LinearProgramFrame),
        ]

        self.sidebar_buttons = []
        self.frames = {}
        self.active_name = None

        for name, frame_cls in self.modules:
            btn = tk.Label(self.sidebar, text=f"  {name}", bg=SIDEBAR_BG, fg=SIDEBAR_FG,
                           font=(FONT_FAMILY, 11), anchor="w", padx=12, pady=8, cursor="hand2")
            btn.pack(fill=tk.X)
            btn.bind("<Button-1>", lambda e, n=name: self.show_frame(n))
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=SIDEBAR_HOVER) if b != self._active_btn() else None)
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg=SIDEBAR_BG) if b != self._active_btn() else None)
            self.sidebar_buttons.append((name, btn))

            frame = frame_cls(self.main_frame)
            self.frames[name] = frame

        # Exit button at bottom
        tk.Frame(self.sidebar, bg=SIDEBAR_BG).pack(fill=tk.BOTH, expand=True)
        exit_btn = tk.Label(self.sidebar, text="  Exit", bg=SIDEBAR_BG, fg="#aaa",
                            font=(FONT_FAMILY, 11), anchor="w", padx=12, pady=8, cursor="hand2")
        exit_btn.pack(fill=tk.X, side=tk.BOTTOM, pady=(0, 10))
        exit_btn.bind("<Button-1>", lambda e: self.destroy())

        self.show_frame("Classification")
        self.bind("<Control-Return>", self._run_active)

    def _active_btn(self):
        for name, btn in self.sidebar_buttons:
            if name == self.active_name:
                return btn
        return None

    def _run_active(self, event=None):
        if self.active_name and self.active_name in self.frames:
            self.frames[self.active_name].run()

    def show_frame(self, name):
        for n, btn in self.sidebar_buttons:
            if n == name:
                btn.configure(bg=SIDEBAR_ACTIVE, fg="white")
            else:
                btn.configure(bg=SIDEBAR_BG, fg=SIDEBAR_FG)
        for n, f in self.frames.items():
            f.pack_forget()
        self.frames[name].pack(fill=tk.BOTH, expand=True, padx=16, pady=10)
        self.active_name = name
