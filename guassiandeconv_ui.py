import os
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from guassiandeconv import run_deblur_pipeline


class DeblurUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gaussian Deblur Runner")
        self.root.geometry("760x560")

        self.sigma_var = tk.StringVar(value="0.01")
        self.lam_var = tk.StringVar(value="0.01")
        self.k_size_var = tk.StringVar(value="21")
        self.k_sigma_var = tk.StringVar(value="2.5")
        self.outdir_var = tk.StringVar(value="outputs")
        self.compare_raw_var = tk.BooleanVar(value=True)

        self.run_button = None
        self.log_box = None
        self._build_ui()

    def _build_ui(self):
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="sigma").grid(row=0, column=0, sticky="w", pady=4)
        tk.Entry(frame, textvariable=self.sigma_var, width=20).grid(row=0, column=1, sticky="w", pady=4)

        tk.Label(frame, text="lambda").grid(row=1, column=0, sticky="w", pady=4)
        tk.Entry(frame, textvariable=self.lam_var, width=20).grid(row=1, column=1, sticky="w", pady=4)

        tk.Label(frame, text="k_size (odd)").grid(row=2, column=0, sticky="w", pady=4)
        tk.Entry(frame, textvariable=self.k_size_var, width=20).grid(row=2, column=1, sticky="w", pady=4)

        tk.Label(frame, text="k_sigma").grid(row=3, column=0, sticky="w", pady=4)
        tk.Entry(frame, textvariable=self.k_sigma_var, width=20).grid(row=3, column=1, sticky="w", pady=4)

        tk.Label(frame, text="outdir").grid(row=4, column=0, sticky="w", pady=4)
        tk.Entry(frame, textvariable=self.outdir_var, width=48).grid(row=4, column=1, sticky="w", pady=4)
        tk.Button(frame, text="Browse", command=self._pick_outdir).grid(row=4, column=2, sticky="w", padx=6)

        tk.Checkbutton(
            frame,
            text="compare raw vs non-raw convolution",
            variable=self.compare_raw_var,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=6)

        self.run_button = tk.Button(frame, text="Run", width=16, command=self._run_clicked)
        self.run_button.grid(row=6, column=0, sticky="w", pady=6)

        tk.Button(frame, text="Clear Log", width=16, command=self._clear_log).grid(row=6, column=1, sticky="w", pady=6)

        self.log_box = ScrolledText(frame, height=22, width=88, state=tk.NORMAL)
        self.log_box.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=8)

        frame.grid_rowconfigure(7, weight=1)
        frame.grid_columnconfigure(1, weight=1)

    def _pick_outdir(self):
        path = filedialog.askdirectory(initialdir=os.getcwd())
        if path:
            self.outdir_var.set(path)

    def _clear_log(self):
        self.log_box.delete("1.0", tk.END)

    def _log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)
        self.root.update_idletasks()

    def _set_running(self, running):
        self.run_button.config(state=tk.DISABLED if running else tk.NORMAL)

    def _validate_inputs(self):
        try:
            sigma = float(self.sigma_var.get())
            lam = float(self.lam_var.get())
            k_size = int(self.k_size_var.get())
            k_sigma = float(self.k_sigma_var.get())
        except ValueError:
            raise ValueError("sigma/lambda/k_size/k_sigma must be numeric.")

        if sigma < 0:
            raise ValueError("sigma must be >= 0.")
        if lam < 0:
            raise ValueError("lambda must be >= 0.")
        if k_size <= 0 or k_size % 2 == 0:
            raise ValueError("k_size must be a positive odd integer.")
        if k_sigma <= 0:
            raise ValueError("k_sigma must be > 0.")

        outdir = self.outdir_var.get().strip()
        if not outdir:
            raise ValueError("outdir cannot be empty.")

        return sigma, lam, k_size, k_sigma, outdir, bool(self.compare_raw_var.get())

    def _run_clicked(self):
        try:
            args = self._validate_inputs()
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return

        self._set_running(True)
        self._log("Starting run...")
        worker = threading.Thread(target=self._run_pipeline, args=args, daemon=True)
        worker.start()

    def _run_pipeline(self, sigma, lam, k_size, k_sigma, outdir, compare_raw):
        try:
            run_deblur_pipeline(
                sigma=sigma,
                lam=lam,
                k_size=k_size,
                k_sigma=k_sigma,
                outdir=outdir,
                compare_raw=compare_raw,
                log_fn=self._log,
            )
        except Exception:
            self._log("Run failed:")
            self._log(traceback.format_exc())
        finally:
            self.root.after(0, lambda: self._set_running(False))


def main():
    root = tk.Tk()
    app = DeblurUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
