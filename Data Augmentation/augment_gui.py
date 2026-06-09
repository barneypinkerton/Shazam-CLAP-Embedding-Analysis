#!/usr/bin/env python3
"""GUI entry point for audio data augmentation."""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from augment_core import AugSource, run_augmentation


class AugmentGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Data Augmentation Tool")
        self.geometry("720x780")
        self.minsize(650, 680)
        self.configure(bg="#1e1e2e")

        self._cancelled = False
        self._running = False
        self._aug_entries = []  # list of (display_str, AugSource)

        self._setup_styles()
        self._build_ui()

        # Pre-populate noise sources
        script_dir = os.path.dirname(os.path.abspath(__file__))
        crowd_path = os.path.join(script_dir, "crowd noise.wav")
        street_path = os.path.join(script_dir, "street noise.wav")

        self._add_entry(AugSource(name="white_noise", kind="white"))
        if os.path.exists(crowd_path):
            self._add_entry(AugSource(name="crowd_noise", kind="file", path=crowd_path))
        if os.path.exists(street_path):
            self._add_entry(AugSource(name="street_noise", kind="file", path=street_path))

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        bg = "#1e1e2e"
        fg = "#cdd6f4"
        accent = "#89b4fa"
        surface = "#313244"
        green = "#a6e3a1"
        red = "#f38ba8"
        subtext = "#a6adc8"

        self._colors = dict(bg=bg, fg=fg, accent=accent, surface=surface,
                            green=green, red=red, subtext=subtext)

        style.configure(".", background=bg, foreground=fg, fieldbackground=surface,
                         borderwidth=0, font=("Helvetica", 12))
        style.configure("TLabel", background=bg, foreground=fg, font=("Helvetica", 12))
        style.configure("Header.TLabel", background=bg, foreground=fg, font=("Helvetica", 14, "bold"))
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

    def _build_ui(self):
        c = self._colors
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

        self.aug_list = tk.Listbox(aug_frame, bg=c["surface"], fg=c["fg"],
                                    selectbackground=c["accent"], selectforeground=c["bg"],
                                    font=("Helvetica", 11), height=5,
                                    borderwidth=0, highlightthickness=1, highlightcolor=c["accent"])
        self.aug_list.pack(fill="both", expand=True, padx=4, pady=4)

        btn_row = ttk.Frame(aug_frame)
        btn_row.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_row, text="+ White Noise", command=self._add_white).pack(side="left", padx=4)
        ttk.Button(btn_row, text="+ Noise File...", command=self._add_file_noise).pack(side="left", padx=4)
        ttk.Button(btn_row, text="+ Pitch Up", command=self._add_pitch_up).pack(side="left", padx=4)
        ttk.Button(btn_row, text="+ Pitch Down", command=self._add_pitch_down).pack(side="left", padx=4)
        ttk.Button(btn_row, text="+ Lo-Fi", command=self._add_lofi).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Remove Selected", style="Remove.TButton",
                    command=self._remove_aug).pack(side="right", padx=4)

        # === Parameters ===
        param_frame = ttk.LabelFrame(self, text="  Parameters  ", padding=10)
        param_frame.pack(fill="x", padx=16, pady=8)

        # Levels
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

        # Snippet duration
        snip_row = ttk.Frame(param_frame)
        snip_row.pack(fill="x", pady=2)
        ttk.Label(snip_row, text="Snippet Duration (s)", width=18, anchor="w").pack(side="left")
        self.snippet_var = tk.StringVar(value="30")
        ttk.Entry(snip_row, textvariable=self.snippet_var, width=8).pack(side="left", padx=(4, 8))
        ttk.Label(snip_row, text="for file-based noise", style="Sub.TLabel").pack(side="left")

        # Seed
        seed_row = ttk.Frame(param_frame)
        seed_row.pack(fill="x", pady=2)
        ttk.Label(seed_row, text="Random Seed", width=18, anchor="w").pack(side="left")
        self.seed_var = tk.StringVar(value="42")
        ttk.Entry(seed_row, textvariable=self.seed_var, width=8).pack(side="left", padx=(4, 8))
        ttk.Label(seed_row, text="leave blank for random", style="Sub.TLabel").pack(side="left")

        # Workers
        workers_row = ttk.Frame(param_frame)
        workers_row.pack(fill="x", pady=2)
        ttk.Label(workers_row, text="Workers", width=18, anchor="w").pack(side="left")
        self.workers_var = tk.StringVar(value="4")
        ttk.Entry(workers_row, textvariable=self.workers_var, width=8).pack(side="left", padx=(4, 8))
        ttk.Label(workers_row, text="parallel CPU cores", style="Sub.TLabel").pack(side="left")

        # === Progress ===
        progress_frame = ttk.LabelFrame(self, text="  Progress  ", padding=10)
        progress_frame.pack(fill="x", padx=16, pady=8)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var, style="Sub.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(progress_frame, style="green.Horizontal.TProgressbar",
                                         length=400, mode="determinate", value=0)
        self.progress.pack(fill="x", pady=(4, 0))

        self.log = tk.Text(progress_frame, bg=c["surface"], fg=c["subtext"], font=("Menlo", 10),
                            height=4, borderwidth=0, highlightthickness=0, state="disabled")
        self.log.pack(fill="x", pady=(8, 0))

        ttk.Label(progress_frame,
                  text="Already-augmented files are skipped — safe to re-run with new sources or levels.",
                  style="Sub.TLabel").pack(anchor="w", pady=(4, 0))

        # === Controls ===
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill="x", padx=16, pady=(8, 16))
        self.start_btn = ttk.Button(ctrl_frame, text="Start Augmentation",
                                     style="Accent.TButton", command=self._start)
        self.start_btn.pack(side="right", padx=4)
        self.cancel_btn = ttk.Button(ctrl_frame, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="right", padx=4)

    def _browse_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _add_entry(self, src: AugSource):
        if src.kind == "white":
            display = "  White Noise  (generated)"
        elif src.kind == "pitch_shift_up":
            display = "  Pitch Shift Up"
        elif src.kind == "pitch_shift_down":
            display = "  Pitch Shift Down"
        elif src.kind == "lofi":
            display = "  Lo-Fi Filter"
        else:
            display = f"  {src.name} — {os.path.basename(src.path)}"
        self._aug_entries.append((display, src))
        self.aug_list.insert(tk.END, display)

    def _add_white(self):
        self._add_entry(AugSource(name="white_noise", kind="white"))

    def _add_pitch_up(self):
        self._add_entry(AugSource(name="pitch_shift_up", kind="pitch_shift_up"))

    def _add_pitch_down(self):
        self._add_entry(AugSource(name="pitch_shift_down", kind="pitch_shift_down"))

    def _add_lofi(self):
        self._add_entry(AugSource(name="lofi", kind="lofi"))

    def _add_file_noise(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if path:
            name = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
            self._add_entry(AugSource(name=name, kind="file", path=path))

    def _remove_aug(self):
        sel = self.aug_list.curselection()
        if sel:
            idx = sel[0]
            self.aug_list.delete(idx)
            self._aug_entries.pop(idx)

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _cancel(self):
        self._cancelled = True
        self.status_var.set("Cancelling...")

    def _start(self):
        if not self._aug_entries:
            messagebox.showerror("Error", "Add at least one augmentation source.")
            return

        try:
            levels = [float(x.strip()) for x in self.levels_var.get().split(",")]
        except ValueError:
            messagebox.showerror("Error", "Levels must be comma-separated numbers.")
            return

        try:
            snippet_dur = float(self.snippet_var.get())
        except ValueError:
            messagebox.showerror("Error", "Snippet duration must be a number.")
            return

        seed = None
        if self.seed_var.get().strip():
            try:
                seed = int(self.seed_var.get())
            except ValueError:
                messagebox.showerror("Error", "Seed must be an integer.")
                return

        try:
            workers = int(self.workers_var.get())
        except ValueError:
            messagebox.showerror("Error", "Workers must be an integer.")
            return

        sources = [src for _, src in self._aug_entries]
        input_dir = self.input_var.get()
        output_dir = self.output_var.get()

        if not os.path.isdir(input_dir):
            messagebox.showerror("Error", f"Input directory not found:\n{input_dir}")
            return

        self._cancelled = False
        self._running = True
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress["value"] = 0
        self.status_var.set("Starting...")
        self._log(f"Sources: {', '.join(src.name for src in sources)}")
        self._log(f"Levels: {levels}")

        def on_progress(completed, total, last_file):
            self.after(0, self._update_progress, completed, total, last_file)

        def run():
            try:
                count = run_augmentation(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    sources=sources,
                    levels=levels,
                    snippet_duration=snippet_dur,
                    seed=seed,
                    workers=workers,
                    progress_callback=on_progress,
                    cancel_check=lambda: self._cancelled,
                )
                self.after(0, self._on_done, count)
            except Exception as e:
                self.after(0, self._on_error, str(e))

        threading.Thread(target=run, daemon=True).start()

    def _update_progress(self, completed, total, last_file):
        pct = (completed / total * 100) if total > 0 else 0
        self.progress["value"] = pct
        fname = os.path.basename(last_file)
        self.status_var.set(f"{completed}/{total} — {fname}")

    def _on_done(self, count):
        self._running = False
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        if self._cancelled:
            self.status_var.set(f"Cancelled. {count} files written.")
            self._log(f"Cancelled after {count} files.")
        else:
            self.progress["value"] = 100
            self.status_var.set(f"Done! {count} files written.")
            self._log(f"Complete. {count} files written.")

    def _on_error(self, msg):
        self._running = False
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.status_var.set("Error")
        self._log(f"ERROR: {msg}")
        messagebox.showerror("Error", msg)


if __name__ == "__main__":
    app = AugmentGUI()
    app.mainloop()
