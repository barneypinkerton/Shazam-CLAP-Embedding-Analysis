#!/usr/bin/env python3
"""Preview of the Data Augmentation GUI layout (no backend logic)."""

import tkinter as tk
from tkinter import ttk, filedialog


class AugmentGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Data Augmentation Tool")
        self.geometry("720x780")
        self.minsize(650, 680)
        self.configure(bg="#1e1e2e")

        style = ttk.Style(self)
        style.theme_use("clam")

        bg = "#1e1e2e"
        fg = "#cdd6f4"
        accent = "#89b4fa"
        surface = "#313244"
        green = "#a6e3a1"
        red = "#f38ba8"
        subtext = "#a6adc8"

        style.configure(".", background=bg, foreground=fg, fieldbackground=surface,
                         borderwidth=0, font=("Helvetica", 12))
        style.configure("TLabel", background=bg, foreground=fg, font=("Helvetica", 12))
        style.configure("Sub.TLabel", background=bg, foreground=subtext, font=("Helvetica", 10))
        style.configure("TEntry", fieldbackground=surface, foreground=fg, insertcolor=fg)
        style.configure("TButton", background=surface, foreground=fg, padding=(12, 6),
                         font=("Helvetica", 11))
        style.map("TButton", background=[("active", accent)], foreground=[("active", bg)])
        style.configure("Accent.TButton", background=accent, foreground=bg, padding=(16, 8),
                         font=("Helvetica", 12, "bold"))
        style.map("Accent.TButton", background=[("active", green)])
        style.configure("Remove.TButton", background=red, foreground=bg, padding=(6, 2),
                         font=("Helvetica", 10))
        style.map("Remove.TButton", background=[("active", "#eb6f92")])
        style.configure("TLabelframe", background=bg, foreground=accent, font=("Helvetica", 12, "bold"))
        style.configure("TLabelframe.Label", background=bg, foreground=accent)
        style.configure("green.Horizontal.TProgressbar", troughcolor=surface, background=green)

        pad = {"padx": 12, "pady": 4}

        # === Directories ===
        dir_frame = ttk.LabelFrame(self, text="  Directories  ", padding=10)
        dir_frame.pack(fill="x", padx=16, pady=(16, 8))

        ttk.Label(dir_frame, text="Input Directory").grid(row=0, column=0, sticky="w", **pad)
        self.input_var = tk.StringVar(value="/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_original")
        ttk.Entry(dir_frame, textvariable=self.input_var, width=48).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(dir_frame, text="Browse", command=lambda: self._browse_dir(self.input_var)).grid(row=0, column=2, **pad)

        ttk.Label(dir_frame, text="Output Directory").grid(row=1, column=0, sticky="w", **pad)
        self.output_var = tk.StringVar(value="/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_augmented")
        ttk.Entry(dir_frame, textvariable=self.output_var, width=48).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(dir_frame, text="Browse", command=lambda: self._browse_dir(self.output_var)).grid(row=1, column=2, **pad)

        dir_frame.columnconfigure(1, weight=1)

        # === Augmentation Sources ===
        aug_frame = ttk.LabelFrame(self, text="  Augmentation Sources  ", padding=10)
        aug_frame.pack(fill="both", expand=True, padx=16, pady=8)

        self.aug_list = tk.Listbox(aug_frame, bg=surface, fg=fg, selectbackground=accent,
                                    selectforeground=bg, font=("Helvetica", 11), height=5,
                                    borderwidth=0, highlightthickness=1, highlightcolor=accent)
        self.aug_list.pack(fill="both", expand=True, padx=4, pady=4)

        # Pre-populate with demo entries
        for entry in [
            "  White Noise  (generated)",
            "  Crowd Noise  — crowd noise.wav",
            "  Street Noise — street noise.wav",
            "  Pitch Shift Up",
            "  Pitch Shift Down",
            "  Lo-Fi Filter",
        ]:
            self.aug_list.insert(tk.END, entry)

        btn_row = ttk.Frame(aug_frame)
        btn_row.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_row, text="+ White Noise", command=self._noop).pack(side="left", padx=4)
        ttk.Button(btn_row, text="+ Noise File...", command=self._noop).pack(side="left", padx=4)
        ttk.Button(btn_row, text="+ Pitch Up", command=self._noop).pack(side="left", padx=4)
        ttk.Button(btn_row, text="+ Pitch Down", command=self._noop).pack(side="left", padx=4)
        ttk.Button(btn_row, text="+ Lo-Fi", command=self._noop).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Remove Selected", style="Remove.TButton",
                    command=self._noop).pack(side="right", padx=4)

        # === Parameters ===
        param_frame = ttk.LabelFrame(self, text="  Parameters  ", padding=10)
        param_frame.pack(fill="x", padx=16, pady=8)

        levels_row = ttk.Frame(param_frame)
        levels_row.pack(fill="x", pady=2)
        ttk.Label(levels_row, text="Levels", width=18, anchor="w").pack(side="left")
        self.levels_var = tk.StringVar(value="20, 10, 0")
        ttk.Entry(levels_row, textvariable=self.levels_var, width=20).pack(side="left", padx=(4, 8))
        ttk.Label(levels_row, text="comma-separated", style="Sub.TLabel").pack(side="left")

        levels_info = (
            "Noise: SNR in dB (20 = quiet, 10 = noticeable, 0 = equal power)  |  "
            "Pitch: semitones (1, 2, 3)  |  Lo-Fi: severity 1/2/3"
        )
        ttk.Label(param_frame, text=levels_info, style="Sub.TLabel").pack(anchor="w", padx=(2, 0), pady=(0, 6))

        snip_row = ttk.Frame(param_frame)
        snip_row.pack(fill="x", pady=2)
        ttk.Label(snip_row, text="Snippet Duration (s)", width=18, anchor="w").pack(side="left")
        ttk.Entry(snip_row, textvariable=tk.StringVar(value="30"), width=8).pack(side="left", padx=(4, 8))
        ttk.Label(snip_row, text="for file-based noise", style="Sub.TLabel").pack(side="left")

        seed_row = ttk.Frame(param_frame)
        seed_row.pack(fill="x", pady=2)
        ttk.Label(seed_row, text="Random Seed", width=18, anchor="w").pack(side="left")
        ttk.Entry(seed_row, textvariable=tk.StringVar(value="42"), width=8).pack(side="left", padx=(4, 8))
        ttk.Label(seed_row, text="leave blank for random", style="Sub.TLabel").pack(side="left")

        workers_row = ttk.Frame(param_frame)
        workers_row.pack(fill="x", pady=2)
        ttk.Label(workers_row, text="Workers", width=18, anchor="w").pack(side="left")
        ttk.Entry(workers_row, textvariable=tk.StringVar(value="4"), width=8).pack(side="left", padx=(4, 8))
        ttk.Label(workers_row, text="parallel CPU cores", style="Sub.TLabel").pack(side="left")

        # === Progress ===
        progress_frame = ttk.LabelFrame(self, text="  Progress  ", padding=10)
        progress_frame.pack(fill="x", padx=16, pady=8)

        ttk.Label(progress_frame, text="450/9000 — blues.00045.wav", style="Sub.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(progress_frame, style="green.Horizontal.TProgressbar",
                                         length=400, mode="determinate", value=35)
        self.progress.pack(fill="x", pady=(4, 0))

        self.log = tk.Text(progress_frame, bg=surface, fg=subtext, font=("Menlo", 10),
                            height=4, borderwidth=0, highlightthickness=0, state="disabled")
        self.log.pack(fill="x", pady=(8, 0))
        self._log("Sources: white_noise, crowd_noise, street_noise, pitch_shift_up, pitch_shift_down, lofi")
        self._log("Levels: [20.0, 10.0, 0.0]")

        # === Controls ===
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill="x", padx=16, pady=(8, 16))
        ttk.Button(ctrl_frame, text="Start Augmentation", style="Accent.TButton").pack(side="right", padx=4)
        ttk.Button(ctrl_frame, text="Cancel").pack(side="right", padx=4)

    def _browse_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _noop(self):
        pass

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")


if __name__ == "__main__":
    app = AugmentGUI()
    app.mainloop()
