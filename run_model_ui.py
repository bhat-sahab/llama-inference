#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_model_ui.py - Tkinter GUI for run_model.py (same logic, graphical).

Pick model / backend / context / MTP / vision, then RUN the server
(live console) or TEST it (boot, measure pp/tg on a fixed prompt, stop).
Plus GPU reset (PnP), ADLX power profiles, and dry-run.

Usage:
  python run_model_ui.py
  python run_model.py gui      (same thing, via run_model.py)
"""
from __future__ import annotations

import io
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.request
from contextlib import redirect_stdout
from tkinter import messagebox, ttk

import run_model as rm


def _plan_lines(rc: rm.RunConfig) -> list[str]:
    return [
        "==============================================",
        "  Starting llama-server",
        "==============================================",
        f"  Model   : {rc.model.path}",
        f"  Backend : {rc.backend.exe}",
        f"  Context : {rc.ctx}   Batch: {rc.batch}   UBatch: {rc.ubatch}   "
        f"ncpu-moe: {rc.ncpu_moe}   KV: {rc.kv[0]}/{rc.kv[1]}",
        f"  MTP     : {' '.join(rc.spec)}" if rc.spec else "  MTP     : off",
        f"  Vision  : {' '.join(rc.mmproj)}" if rc.mmproj else "  Vision  : off",
        "",
        f"  Server: http://localhost:{rc.port}",
        "==============================================",
        "",
    ]


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("llama-server launcher  (Qwen3.6 / KAT / Qwen3.8)")
        root.geometry("900x640")

        self.q: queue.Queue = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.busy = False

        # ── selection frame ──────────────────────────────────────────
        sel = ttk.LabelFrame(root, text=" Selection ")
        sel.pack(fill="x", padx=8, pady=(8, 2))
        sel.columnconfigure(3, weight=1)

        ttk.Label(sel, text="Model:").grid(row=0, column=0, sticky="w", padx=6, pady=2)
        self.model_var = tk.StringVar()
        self.model_cb = ttk.Combobox(sel, textvariable=self.model_var,
                                     values=[m.label for m in rm.MODELS],
                                     state="readonly", width=52)
        self.model_cb.grid(row=0, column=1, columnspan=3, sticky="we", padx=6, pady=2)
        self.model_cb.current(0)
        self.model_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh())

        ttk.Label(sel, text="Backend:").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        self.backend_var = tk.StringVar()
        self.backend_cb = ttk.Combobox(sel, textvariable=self.backend_var, width=28,
                                       state="readonly")
        self.backend_cb.grid(row=1, column=1, sticky="w", padx=6, pady=2)
        self.backend_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh())

        ttk.Label(sel, text="Context:").grid(row=1, column=2, sticky="e", padx=6, pady=2)
        self.ctx_var = tk.StringVar(value="32768")
        self.ctx_cb = ttk.Combobox(sel, textvariable=self.ctx_var,
                                   values=[str(c) for c in rm.CTX_CHOICES], width=8,
                                   state="readonly")
        self.ctx_cb.grid(row=1, column=3, sticky="w", padx=6, pady=2)

        self.mtp_var = tk.BooleanVar(value=True)
        self.mtp_cb = ttk.Checkbutton(sel, text="MTP", variable=self.mtp_var)
        self.mtp_cb.grid(row=2, column=0, sticky="w", padx=6, pady=2)
        self.mtp_mode_var = tk.StringVar(value="tuned")
        self.mtp_mode_cb = ttk.Combobox(sel, textvariable=self.mtp_mode_var,
                                        values=["normal", "tuned"], width=8,
                                        state="readonly")
        self.mtp_mode_cb.grid(row=2, column=1, sticky="w", padx=6, pady=2)

        self.vis_var = tk.BooleanVar(value=True)
        self.vis_cb = ttk.Checkbutton(sel, text="Vision (mmproj)", variable=self.vis_var)
        self.vis_cb.grid(row=2, column=2, columnspan=2, sticky="w", padx=6, pady=2)

        ttk.Label(sel, text="Port:").grid(row=2, column=4, sticky="e", padx=6, pady=2)
        self.port_var = tk.StringVar(value=str(rm.DEFAULT_PORT))
        ttk.Entry(sel, textvariable=self.port_var, width=6).grid(
            row=2, column=5, sticky="w", padx=6, pady=2)

        self.hint_var = tk.StringVar(value="")
        ttk.Label(sel, textvariable=self.hint_var, foreground="#888888",
                  anchor="w").grid(row=3, column=0, columnspan=6, sticky="we",
                                   padx=6, pady=(4, 0))

        # ── action buttons ───────────────────────────────────────────
        act = ttk.Frame(root)
        act.pack(fill="x", padx=8, pady=2)
        self.btn_run = ttk.Button(act, text="RUN", command=lambda: self.start_run(dry=False))
        self.btn_run.pack(side="left", padx=3)
        self.btn_test = ttk.Button(act, text="TEST (bench)", command=lambda: self.start_run(dry=False, test=True))
        self.btn_test.pack(side="left", padx=3)
        self.btn_stop = ttk.Button(act, text="STOP", command=self.stop_server, state="disabled")
        self.btn_stop.pack(side="left", padx=3)
        self.btn_dry = ttk.Button(act, text="DRY-RUN (print cmd)", command=lambda: self.start_run(dry=True))
        self.btn_dry.pack(side="left", padx=3)

        ttk.Separator(act, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(act, text="GPU Reset (PnP)", command=self.gpu_reset).pack(side="left", padx=3)
        ttk.Label(act, text="Power:").pack(side="left", padx=(6, 2))
        self.power_var = tk.StringVar(value="low")
        self.power_cb = ttk.Combobox(act, textvariable=self.power_var,
                                     values=["low", "stock", "max"], width=6,
                                     state="readonly")
        self.power_cb.pack(side="left")
        ttk.Button(act, text="Apply", command=self.apply_power).pack(side="left", padx=3)
        ttk.Button(act, text="List", command=self.show_list).pack(side="right", padx=3)

        # ── test options ─────────────────────────────────────────────
        tst = ttk.LabelFrame(root, text=" Test options (for TEST) ")
        tst.pack(fill="x", padx=8, pady=2)
        ttk.Label(tst, text="Prompt:").grid(row=0, column=0, sticky="w", padx=6, pady=2)
        self.prompt_var = tk.StringVar(value="Count from 1 to 100.")
        self.prompt_e = ttk.Entry(tst, textvariable=self.prompt_var)
        self.prompt_e.grid(row=0, column=1, columnspan=2, sticky="we", padx=6, pady=2)
        ttk.Label(tst, text="Tokens:").grid(row=0, column=3, sticky="e", padx=6, pady=2)
        self.tokens_var = tk.StringVar(value="64")
        ttk.Entry(tst, textvariable=self.tokens_var, width=6).grid(
            row=0, column=4, sticky="w", padx=6, pady=2)
        tst.columnconfigure(1, weight=1)

        # ── console ──────────────────────────────────────────────────
        con = ttk.LabelFrame(root, text=" Console ")
        con.pack(fill="both", expand=True, padx=8, pady=2)
        btnbar = ttk.Frame(con)
        btnbar.pack(fill="x")
        ttk.Button(btnbar, text="Clear", command=self._console_clear).pack(
            side="right", padx=4, pady=2)
        body = ttk.Frame(con)
        body.pack(fill="both", expand=True)
        self.console = tk.Text(body, state="disabled", wrap="none",
                               font=("Consolas", 9), background="#101010",
                               foreground="#e0e0e0", padx=6, pady=6)
        sb = ttk.Scrollbar(body, command=self.console.yview)
        self.console.configure(yscrollcommand=sb.set)
        self.console.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # ── status bar ───────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Idle — pick a model and press RUN or TEST.")
        ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w",
                  background="#f0f0f0").pack(fill="x", padx=8, pady=(2, 6))

        self.refresh()
        self._append("  Tip: DRY-RUN prints the exact llama-server command without starting it.\n")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._poll)

    # ── selection helpers ────────────────────────────────────────────
    def model(self) -> rm.Model:
        try:
            idx = self.model_cb.current()
            return rm.MODELS[idx]
        except Exception:
            return rm.MODELS[0]

    def backend(self) -> rm.Backend | None:
        bs = rm.backends_for(self.model())
        try:
            i = self.backend_cb.current()
            return bs[i]
        except Exception:
            return None

    def ctx(self) -> int:
        try:
            return int(self.ctx_var.get())
        except ValueError:
            return 32768

    def port(self) -> int:
        try:
            return int(self.port_var.get().strip())
        except ValueError:
            return rm.DEFAULT_PORT

    def tokens(self) -> int:
        try:
            return max(1, int(self.tokens_var.get().strip()))
        except ValueError:
            return 64

    def rc(self) -> rm.RunConfig:
        m, b = self.model(), self.backend()
        return rm.resolve(m, b, self.ctx(), self.mtp_var.get(),
                          self.vis_var.get(), self.port(), self.mtp_mode_var.get())

    def refresh(self):
        """Re-sync backend list + MTP/vision validity when selection changes."""
        m = self.model()
        bs = rm.backends_for(m)
        keys = [b.key for b in bs]
        self.backend_cb["values"] = keys
        cur = self.backend_var.get()
        if cur in keys:
            self.backend_cb.current(keys.index(cur))
        else:
            self.backend_cb.current(0)
            self.backend_var.set(bs[0].key)

        mtp_was_enabled = self.mtp_cb.state() == "normal"
        mtp_ok = self.backend() is not None and rm.mtp_allowed(m, self.backend())
        self.mtp_cb.config(state="normal" if mtp_ok else "disabled")
        self.mtp_mode_cb.config(state="readonly" if mtp_ok else "disabled")
        if not mtp_ok:
            self.mtp_var.set(False)
        elif not mtp_was_enabled:
            self.mtp_var.set(True)  # auto-on when MTP becomes valid (bat default)

        vis_was_enabled = self.vis_cb.state() == "normal"
        vis_ok = bool(m.mmproj) and self.backend() is not None \
            and self.backend().key != "stew" and m.mmproj.exists()
        self.vis_cb.config(state="normal" if vis_ok else "disabled")
        if not vis_ok:
            self.vis_var.set(False)
        elif not vis_was_enabled:
            self.vis_var.set(True)  # auto-on when vision becomes valid

        pp, tg = m.speeds.get(self.ctx(), ("?", "?"))
        hint = f"expected: pp {pp} | tg {tg} t/s"
        if m.key == "q38m" and self.ctx() == 262144:
            hint = "WARNING: Q3_K_M 256K broken (VRAM compute buffers)"
        if m.key == "q38" and self.ctx() == 262144:
            hint = "WARNING: IQ4_XS 256K not supported"
        self.hint_var.set(hint)
        self._hint_managed()

    def _hint_managed(self):
        """Show the expected-speed hint in the status bar when idle."""
        if not self.busy:
            self.status_var.set(
                f"{self.model().key} | backend {self.backend_var.get()} | "
                f"ctx {self.ctx()} | {self.hint_var.get()}")

    # ── console ──────────────────────────────────────────────────────
    def _append(self, text: str):
        self.console.config(state="normal")
        self.console.insert("end", text)
        self.console.config(state="disabled")
        self.console.see("end")

    def _console_clear(self):
        self.console.config(state="normal")
        self.console.delete("1.0", "end")
        self.console.config(state="disabled")

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._append(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "done":
                    self.busy = False
                    self.proc = None
                    self.btn_run.config(state="normal")
                    self.btn_test.config(state="normal")
                    self.btn_stop.config(state="disabled")
                    self.model_cb.config(state="readonly")
                    self.backend_cb.config(state="readonly")
                    self.ctx_cb.config(state="readonly")
                    self._hint_managed()
                    if payload:
                        self.status_var.set(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ── run / test ───────────────────────────────────────────────────
    def start_run(self, dry: bool = False, test: bool = False):
        if self.busy:
            return
        rc = self.rc()
        if dry:
            for ln in _plan_lines(rc):
                self._append(ln + "\n")
            self._append("  Command line:\n  " + " ".join(
                f'"{x}"' if " " in x else x for x in rm.build_cmd(rc)) + "\n\n")
            self.status_var.set("Dry-run — command shown, nothing started.")
            return
        self.busy = True
        self.proc = None
        self.btn_run.config(state="disabled")
        self.btn_test.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.model_cb.config(state="disabled")
        self.backend_cb.config(state="disabled")
        self.ctx_cb.config(state="disabled")
        # capture test inputs on the main thread (Tcl isn't thread-safe)
        prompt = self.prompt_var.get().strip() or "Count from 1 to 100."
        n_tokens = self.tokens()
        threading.Thread(target=self._job,
                         args=(rc, "test" if test else "run", prompt, n_tokens),
                         daemon=True).start()

    def _job(self, rc: rm.RunConfig, mode: str, prompt: str = "", n_tokens: int = 64):
        q = self.q
        q.put(("status", f"Killing stale process on port {rc.port}, launching server..."))
        try:
            rm.kill_port(rc.port)
        except Exception:
            pass
        for ln in _plan_lines(rc):
            q.put(("log", ln + "\n"))

        cmd = rm.build_cmd(rc)
        kw = dict(env=rm.launch_env(rc), stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT, text=True,
                  encoding="utf-8", errors="replace")
        if rm.IS_WIN:
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            proc = subprocess.Popen(cmd, **kw)
        except OSError as e:
            q.put(("log", f"  Failed to start: {e}\n"))
            q.put(("done", "Failed to start server."))
            return
        self.proc = proc

        def reader():
            try:
                for line in proc.stdout:
                    q.put(("log", line))
            finally:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
        threading.Thread(target=reader, daemon=True).start()

        if mode == "test":
            self._test_job(proc, rc, q, prompt, n_tokens)
        else:
            code = proc.wait()
            q.put(("log", f"\nServer stopped (code {code}).\n"))
            q.put(("done", f"Server stopped (code {code})."))

    def _test_job(self, proc: subprocess.Popen, rc: rm.RunConfig, q: queue.Queue,
                  prompt: str = "Count from 1 to 100.", n_tokens: int = 64):
        base = f"http://localhost:{rc.port}"
        q.put(("status", "Waiting for server to load..."))
        t0 = time.time()
        ready, alive = False, True
        while True:
            if proc.poll() is not None:
                alive = False
                q.put(("log", f"\nServer exited early (code {proc.returncode}).\n"))
                break
            try:
                urllib.request.urlopen(base + "/", timeout=2).read()
                ready = True
                break
            except Exception:
                pass
            el = int(time.time() - t0)
            if el >= 300:
                q.put(("status", "Server never became ready — stopping."))
                self._kill(proc)
                break
            q.put(("status", f"Waiting for server to load... {el}s"))
            time.sleep(1.5)

        if not alive:
            q.put(("done", f"Server exited early (code {proc.returncode})."))
            return
        if not ready:
            q.put(("done", "Server never became ready."))
            return

        q.put(("status", "Server ready — running prompt test..."))
        q.put(("log", f"  Test: {n_tokens} tokens, prompt: {prompt!r}\n"))
        try:
            res = rm.http_json(base + "/completion",
                               {"prompt": prompt, "n_tokens": n_tokens,
                                "stream": False, "temperature": 0.0},
                               timeout=300)
        except Exception as e:
            q.put(("log", f"  Test request failed: {e}\n"))
            self._kill(proc)
            q.put(("done", "Test request failed."))
            return

        t = res.get("timings", {}) or {}
        pp_tok, pp_t = t.get("tokens_prompt"), t.get("t_prompt_process")
        tg_tok, tg_t = t.get("tokens_predicted"), t.get("t_predict")
        pp = pp_tok / pp_t if pp_tok and pp_t else None
        tg = tg_tok / tg_t if tg_tok and tg_t else None
        exp_pp, exp_tg = rc.model.speeds.get(rc.ctx, ("?", "?"))
        out = (res.get("content") or res.get("content_str") or "").strip()

        block = (
            "  ── Measured ─────────────────────────────\n"
            + (f"  prompt : {pp_t:.2f} s for {pp_tok} tok -> {pp:6.1f} t/s\n" if pp else "  prompt : n/a\n")
            + (f"  decode : {tg_t:.2f} s for {tg_tok} tok -> {tg:6.1f} t/s\n" if tg else "  decode : n/a\n")
            + f"  ── Expected {rc.ctx} ──────────────────────\n"
            + f"  pp {exp_pp} t/s | tg {exp_tg} t/s  (launcher table)\n"
            + "  ── Output ───────────────────────────────\n"
            + "  " + out + "\n"
        )
        q.put(("log", block))
        q.put(("status", "Stopping server..."))
        self._kill(proc)
        q.put(("done", "Test done."))

    def _kill(self, proc: subprocess.Popen):
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        except Exception:
            pass

    def stop_server(self):
        p = self.proc
        if p and p.poll() is None:
            self.status_var.set("Stopping server...")
            threading.Thread(target=self._kill, args=(p,), daemon=True).start()

    # ── GPU tools ────────────────────────────────────────────────────
    def gpu_reset(self):
        if self.busy:
            return
        self.status_var.set("GPU reset (PnP cycle) — screen goes black briefly...")
        threading.Thread(target=self._tool,
                         args=(rm.gpu_reset, "GPU reset done."), daemon=True).start()

    def apply_power(self):
        if self.busy:
            return
        prof = self.power_var.get()
        threading.Thread(target=self._tool,
                         args=(lambda: rm.power_profile(prof),
                               f"Power profile {prof} applied."),
                         daemon=True).start()

    def _tool(self, fn, done_msg: str):
        q = self.q
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                fn()
        except Exception as e:
            buf.write(str(e) + "\n")
        for ln in buf.getvalue().splitlines():
            q.put(("log", ln + "\n"))
        q.put(("log", "\n"))
        q.put(("done", done_msg))

    def show_list(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rm.list_all()
        self._append(buf.getvalue() + "\n")

    # ── lifecycle ────────────────────────────────────────────────────
    def on_close(self):
        if self.busy and self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("Exit", "Server is still running. Stop it and exit?"):
                return
            self._kill(self.proc)
        self.root.destroy()


def main():
    rm._setup_console()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
