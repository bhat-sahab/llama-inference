#!/usr/bin/env python3
"""
llama.cpp Model Launcher — GUI for RX 9070 XT + i9-14900K
Zero-dependency: uses tkinter (bundled with Python on Windows).

Usage:  python launcher.py
"""

import json
import os
import subprocess
import sys
import struct
import threading
import urllib.request
from pathlib import Path
from tkinter import Tk, Toplevel, Frame, Label, Button, Entry, Spinbox, Checkbutton
from tkinter import StringVar, IntVar, BooleanVar, Text, Scrollbar
from tkinter import ttk, messagebox, filedialog
from tkinter import HORIZONTAL, VERTICAL, BOTH, LEFT, RIGHT, TOP, BOTTOM, X, Y, W, E, N, S, END, WORD, DISABLED, NORMAL

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DIST_DIR = SCRIPT_DIR / "llama.cpp"
CONFIG_FILE = SCRIPT_DIR / "launcher_config.json"
CUSTOM_MODELS_FILE = SCRIPT_DIR / "custom_models.json"

MODELS = {
    "Gemma-4-12B (Q4_K_M, 6.7 GB)": {
        "key": "Gemma",
        "path": "~/.lmstudio/models/unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf",
        "ctx": 32000, "batch": 8192, "ubatch": 4096,
        "speed": "~60 t/s", "template": None, "jinja": False,
        "modes": ["Default", "MTP"],
        "mode_overrides": {
            "Default": {"ctx": 32000, "batch": 8192, "ubatch": 4096, "kv_k": "q8_0", "kv_v": "q8_0"},
            "MTP":     {"ctx": 32000, "batch": 8192, "ubatch": 4096, "kv_k": "q8_0", "kv_v": "q8_0"},
        },
        "notes": "Dense model, fits entirely on 16 GB VRAM",
        "hf_repo": "unsloth/gemma-4-12b-it-GGUF",
        "hf_file": "gemma-4-12b-it-Q4_K_M.gguf"
    },
    "Qwen3.5-9B (Q4_K_M, 5.3 GB)": {
        "key": "Qwen35",
        "path": "~/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf",
        "ctx": 128000, "batch": 8192, "ubatch": 8192,
        "speed": "~72 t/s", "template": "chat_template.jinja", "jinja": True,
        "modes": ["Default"],
        "mode_overrides": {
            "Default": {"ctx": 128000, "batch": 8192, "ubatch": 8192, "kv_k": "q8_0", "kv_v": "q8_0"},
        },
        "notes": "Fastest model on this system",
        "hf_repo": "lmstudio-community/Qwen3.5-9B-GGUF",
        "hf_file": "Qwen3.5-9B-Q4_K_M.gguf"
    },
    "Qwen3-Coder-30B-A3B (Q4_K_M, ~18 GB)": {
        "key": "QwenCoder",
        "path": "~/.lmstudio/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        "ctx": 32000, "batch": 4096, "ubatch": 4096,
        "speed": "~43 t/s (default)", "template": "qwen3-coder-30b-template.jinja", "jinja": True,
        "modes": ["Default", "Full (CPU experts)", "LongCtx (200K)"],
        "mode_overrides": {
            "Default":             {"ctx": 32000,  "batch": 4096, "ubatch": 4096, "kv_k": "q8_0", "kv_v": "q8_0"},
            "Full (CPU experts)":  {"ctx": 64000,  "batch": 3072, "ubatch": 3072, "kv_k": "q8_0", "kv_v": "q8_0"},
            "LongCtx (200K)":      {"ctx": 200000, "batch": 3072, "ubatch": 3072, "kv_k": "q4_0", "kv_v": "q4_0"},
        },
        "notes": "MoE, 3B active / 30B total",
        "hf_repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "hf_file": "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
    },
    "Ternary-Bonsai-27B (Q2_0, 7.3 GB)": {
        "key": "Bonsai",
        "path": "~/.lmstudio/models/prism-ml/Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-Q2_g64.gguf",
        "path_fallback": "~/.lmstudio/models/prism-ml/Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-Q2_0.gguf",
        "ctx": 128000, "batch": 2048, "ubatch": 2048,
        "speed": "~58 t/s", "template": "chat_template.jinja", "jinja": True,
        "modes": ["Default", "LongCtx (256K, Q4 KV)", "Thinking", "LongCtx + Thinking"],
        "mode_overrides": {
            "Default":              {"ctx": 128000, "batch": 2048, "ubatch": 2048, "kv_k": "q4_0", "kv_v": "q4_0"},
            "LongCtx (256K, Q4 KV)": {"ctx": 262144, "batch": 2048, "ubatch": 2048, "kv_k": "q4_0", "kv_v": "q4_0"},
            "Thinking":              {"ctx": 128000, "batch": 2048, "ubatch": 2048, "kv_k": "q4_0", "kv_v": "q4_0"},
            "LongCtx + Thinking":    {"ctx": 262144, "batch": 2048, "ubatch": 2048, "kv_k": "q4_0", "kv_v": "q4_0"},
        },
        "notes": "Q2_0 ternary — q4_0 KV for 58 t/s, 256K native context",
        "hf_repo": "prism-ml/Ternary-Bonsai-27B-gguf",
        "hf_file": "Ternary-Bonsai-27B-Q2_g64.gguf"
    },
    "Bonsai-8B (Q1_0, 1.5 GB)": {
        "key": "Bonsai8B",
        "path": "~/.lmstudio/models/prism-ml/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf",
        "ctx": 65536, "batch": 4096, "ubatch": 4096,
        "speed": "~200+ t/s", "template": "chat_template.jinja", "jinja": True,
        "modes": ["Default", "LongCtx (65K)"],
        "mode_overrides": {
            "Default":      {"ctx": 65536, "batch": 4096, "ubatch": 4096, "kv_k": "q8_0", "kv_v": "q8_0"},
            "LongCtx (65K)": {"ctx": 65536, "batch": 4096, "ubatch": 4096, "kv_k": "q8_0", "kv_v": "q8_0"},
        },
        "notes": "1-bit — runs on any backend, only 1.5 GB",
        "hf_repo": "prism-ml/Bonsai-8B-gguf",
        "hf_file": "Bonsai-8B-Q1_0.gguf"
    },
}


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────
def resolve_path(path_str):
    """Resolve ~ to USERPROFILE."""
    return Path(os.path.expandvars(path_str.replace("~", os.environ["USERPROFILE"])))


def find_hip_sdk():
    """Auto-detect ROCm/HIP SDK. Checks pip package, system PATH, and common locations."""
    # 1. Check if HIP is already on PATH
    hip_paths = [p for p in os.environ.get("PATH", "").split(";")
                 if "rocm" in p.lower() or "hip" in p.lower()]
    for p in hip_paths:
        cand = Path(p) / "hipcc.exe"
        if cand.exists():
            return Path(p).parent

    # 2. Check the pip-installed _rocm_sdk_devel package (MS Store Python)
    local_packages = Path(os.environ["USERPROFILE"]) / "AppData/Local/Packages"
    for pkg in sorted(local_packages.glob("PythonSoftwareFoundation.*"), reverse=True):
        for lp in pkg.glob("LocalCache/local-packages/Python*/site-packages/_rocm_sdk_devel"):
            if lp.is_dir():
                return lp

    # 3. Check standard HIP SDK install location
    for prog in (Path(os.environ["PROGRAMFILES"]), Path(os.environ["PROGRAMFILES(X86)"])):
        cand = prog / "AMD" / "ROCm" / "bin"
        if (cand / "hipcc.exe").exists():
            return cand

    return None


def read_gguf_meta(path):
    """Read basic metadata from a GGUF file without loading the full model.
    Returns dict with 'ctx' (n_ctx_train) and 'arch' if available."""
    meta = {"ctx": None, "arch": None}
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return meta
            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n_kv):
                klen = struct.unpack("<I", f.read(4))[0]
                key = f.read(klen).decode("utf-8")
                f.read(4)  # type
                if key == "general.architecture":
                    slen = struct.unpack("<Q", f.read(8))[0]
                    meta["arch"] = f.read(slen).decode("utf-8")
                elif key == "llama.context_length" or key == "llama.max_position_embeddings":
                    f.read(4)  # type (usually 4 = uint32)
                    meta["ctx"] = struct.unpack("<I", f.read(4))[0]
                elif key == "llama.attention.layer_count":
                    f.read(4)
                    meta["n_layer"] = struct.unpack("<I", f.read(4))[0]
                else:
                    # Skip value: read type byte then skip appropriate bytes
                    vtype = key[-1] if not isinstance(key, int) else 0
                    # Simple skip — read 4 bytes for scalar types
                    f.read(4)
    except Exception:
        pass
    return meta


def load_config():
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}


def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def load_custom_models():
    """Load user-defined custom models from JSON file."""
    if CUSTOM_MODELS_FILE.exists():
        try:
            return json.loads(CUSTOM_MODELS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_custom_models(models_dict):
    """Persist custom models to JSON file."""
    # Strip internal state before saving
    clean = {}
    for name, info in models_dict.items():
        if info.get("key") == "Custom":
            clean[name] = {
                "key": "Custom",
                "path": info["path"],
                "ctx": info.get("ctx", 8192),
                "batch": info.get("batch", 4096),
                "ubatch": info.get("ubatch", 1024),
                "speed": "—",
                "template": None,
                "jinja": False,
                "modes": ["Default"],
                "mode_overrides": {"Default": {
                    "ctx": info.get("ctx", 8192),
                    "batch": info.get("batch", 4096),
                    "ubatch": info.get("ubatch", 1024),
                    "kv_k": "q8_0", "kv_v": "q8_0"
                }},
                "notes": f"Custom model: {Path(info['path']).name}"
            }
    CUSTOM_MODELS_FILE.write_text(json.dumps(clean, indent=2))


def check_server_health(port, timeout=3):
    """Check if the llama-server is responding on the given port."""
    import socket
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        sock.close()
        return True
    except (ConnectionRefusedError, OSError):
        pass
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────
class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("llama.cpp Launcher")
        self.root.geometry("720x640")
        self.root.minsize(620, 580)
        self.root.resizable(True, True)

        self.config = load_config()
        self.server_process = None
        self.monitor_thread = None
        self.health_thread = None
        self.running = False

        # Load custom models from disk
        custom = load_custom_models()
        MODELS.update(custom)

        # Configurable models directory
        self.models_dir = self.config.get("models_dir", "~/.lmstudio/models")

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabelframe", padding=8)
        style.configure("TLabelframe.Label", font=("Segoe UI", 9, "bold"))
        style.configure("TButton", padding=(12, 6), font=("Segoe UI", 9))
        style.configure("Launch.TButton", padding=(16, 8), font=("Segoe UI", 10, "bold"))

        # ── Model Selection ──
        top = ttk.Frame(root, padding=(12, 12, 12, 0))
        top.pack(fill=X)

        ttk.Label(top, text="Model", font=("Segoe UI", 10, "bold")).pack(anchor=W)
        self.model_var = StringVar()
        self.model_combo = ttk.Combobox(top, textvariable=self.model_var,
                                         values=list(MODELS.keys()) + ["─" * 30, "📂 Browse for model..."], state="readonly", width=60)
        self.model_combo.pack(fill=X, pady=(4, 2))
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_select)

        self.model_note = ttk.Label(top, text="", foreground="gray")
        self.model_note.pack(anchor=W, pady=(0, 2))

        # ── Backend ──
        backend_frame = ttk.LabelFrame(root, text="Backend", padding=(12, 8))
        backend_frame.pack(fill=X, padx=12, pady=(4, 2))

        self.backend_var = StringVar(value="ROCm")
        ttk.Radiobutton(backend_frame, text="ROCm (PrismML)  (54 t/s for Bonsai)", variable=self.backend_var,
                        value="ROCm", command=self.on_backend_change).pack(anchor=W, pady=2)
        ttk.Radiobutton(backend_frame, text="ROCm (mainline)  (for Gemma/Qwen, no Q2_0)", variable=self.backend_var,
                        value="ROCm-mainline", command=self.on_backend_change).pack(anchor=W, pady=2)
        ttk.Radiobutton(backend_frame, text="Vulkan  (58 t/s for Bonsai, zero setup)", variable=self.backend_var,
                        value="Vulkan", command=self.on_backend_change).pack(anchor=W, pady=2)

        self.hip_status = ttk.Label(backend_frame, text="", foreground="gray")
        self.hip_status.pack(anchor=W, pady=(2, 0))
        self.on_backend_change()

        # ── Mode ──
        mode_frame = ttk.LabelFrame(root, text="Mode", padding=(12, 8))
        mode_frame.pack(fill=X, padx=12, pady=(4, 2))
        self.mode_var = StringVar(value="Default")
        self.mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var,
                                        state="readonly", width=40)
        self.mode_combo.pack(fill=X)
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)

        # ── Quick Settings ──
        settings = ttk.LabelFrame(root, text="Quick Settings", padding=(12, 8))
        settings.pack(fill=X, padx=12, pady=(4, 2))

        row1 = ttk.Frame(settings)
        row1.pack(fill=X, pady=2)
        ttk.Label(row1, text="Context:", width=10).pack(side=LEFT)
        self.ctx_var = IntVar(value=32000)
        self.ctx_spin = ttk.Spinbox(row1, from_=512, to=256000, increment=1024,
                                     textvariable=self.ctx_var, width=10)
        self.ctx_spin.pack(side=LEFT, padx=(0, 20))

        ttk.Label(row1, text="GPU Layers:", width=11).pack(side=LEFT)
        self.ngl_var = IntVar(value=999)
        ttk.Spinbox(row1, from_=0, to=999, increment=1,
                    textvariable=self.ngl_var, width=6).pack(side=LEFT, padx=(0, 20))

        ttk.Label(row1, text="Threads:", width=8).pack(side=LEFT)
        self.threads_var = IntVar(value=6)
        ttk.Spinbox(row1, from_=1, to=32, increment=1,
                    textvariable=self.threads_var, width=5).pack(side=LEFT)

        row2 = ttk.Frame(settings)
        row2.pack(fill=X, pady=2)
        ttk.Label(row2, text="Batch:", width=10).pack(side=LEFT)
        self.batch_var = IntVar(value=4096)
        ttk.Spinbox(row2, from_=64, to=32768, increment=512,
                    textvariable=self.batch_var, width=10).pack(side=LEFT, padx=(0, 20))

        ttk.Label(row2, text="Micro-batch:", width=11).pack(side=LEFT)
        self.ubatch_var = IntVar(value=4096)
        ttk.Spinbox(row2, from_=64, to=32768, increment=512,
                    textvariable=self.ubatch_var, width=10).pack(side=LEFT, padx=(0, 20))

        ttk.Label(row2, text="Port:", width=8).pack(side=LEFT)
        self.port_var = IntVar(value=8081)
        ttk.Spinbox(row2, from_=1024, to=65535, increment=1,
                    textvariable=self.port_var, width=6).pack(side=LEFT)

        row3 = ttk.Frame(settings)
        row3.pack(fill=X, pady=2)
        ttk.Label(row3, text="Template:", width=10).pack(side=LEFT)
        self.template_var = StringVar(value="None")
        self.template_combo = ttk.Combobox(row3, textvariable=self.template_var,
                                            state="readonly", width=40)
        self.template_combo.pack(side=LEFT, fill=X, expand=True)
        self._refresh_templates()
        self.template_combo.bind("<<ComboboxSelected>>", self._on_template_manual)

        # ── Advanced Options (collapsible) ──
        self.adv_frame = ttk.LabelFrame(root, text="Advanced", padding=(12, 8))

        adv_toggle = ttk.Frame(root)
        adv_toggle.pack(fill=X, padx=12, pady=(2, 2))
        self.show_adv = BooleanVar(value=False)
        ttk.Checkbutton(adv_toggle, text="Show advanced options", variable=self.show_adv,
                        command=self.toggle_advanced).pack(anchor=W)

        adv = ttk.Frame(self.adv_frame)
        adv.pack(fill=X)

        self.flash_var = BooleanVar(value=True)
        ttk.Checkbutton(adv, text="Flash Attention", variable=self.flash_var).pack(anchor=W)
        self.mlock_var = BooleanVar(value=True)
        ttk.Checkbutton(adv, text="mlock (lock memory)", variable=self.mlock_var).pack(anchor=W)
        self.nommap_var = BooleanVar(value=True)
        ttk.Checkbutton(adv, text="--no-mmap", variable=self.nommap_var).pack(anchor=W)
        self.jinja_var = BooleanVar(value=False)
        ttk.Checkbutton(adv, text="--jinja (chat template)", variable=self.jinja_var).pack(anchor=W)

        # ── Generation Quality Controls ──
        gen_row = ttk.Frame(adv)
        gen_row.pack(fill=X, pady=(6, 2))

        ttk.Label(gen_row, text="Temperature:", width=12).pack(side=LEFT)
        self.temperature_var = StringVar(value="0.7")
        ttk.Entry(gen_row, textvariable=self.temperature_var, width=8).pack(side=LEFT, padx=(0, 8))

        ttk.Label(gen_row, text="Top P:", width=10).pack(side=LEFT)
        self.top_p_var = StringVar(value="0.95")
        ttk.Entry(gen_row, textvariable=self.top_p_var, width=8).pack(side=LEFT, padx=(0, 8))

        ttk.Label(gen_row, text="Top K:", width=10).pack(side=LEFT)
        self.top_k_var = StringVar(value="20")
        ttk.Entry(gen_row, textvariable=self.top_k_var, width=8).pack(side=LEFT, padx=(0, 8))

        ttk.Label(gen_row, text="Repetition Penalty:", width=18).pack(side=LEFT)
        self.repetition_penalty_var = StringVar(value="1.0")
        ttk.Entry(gen_row, textvariable=self.repetition_penalty_var, width=8).pack(side=LEFT)

        override_row = ttk.Frame(adv)
        override_row.pack(fill=X, pady=(4, 0))
        ttk.Label(override_row, text="Override tensors:").pack(side=LEFT)
        self.override_var = StringVar()
        ttk.Entry(override_row, textvariable=self.override_var, width=50).pack(side=LEFT, padx=(6, 0), fill=X, expand=True)

        extra_row = ttk.Frame(adv)
        extra_row.pack(fill=X, pady=(4, 0))
        ttk.Label(extra_row, text="Extra flags:").pack(side=LEFT)
        self.extra_var = StringVar()
        ttk.Entry(extra_row, textvariable=self.extra_var, width=50).pack(side=LEFT, padx=(6, 0), fill=X, expand=True)

        # ── Buttons ──
        self.btn_frame = ttk.Frame(root, padding=(12, 8, 12, 8))
        self.btn_frame.pack(fill=X, side=BOTTOM)

        self.launch_btn = ttk.Button(self.btn_frame, text="▶  Launch Server",
                                      style="Launch.TButton", command=self.launch)
        self.launch_btn.pack(side=LEFT, padx=(0, 10))

        self.stop_btn = ttk.Button(self.btn_frame, text="■  Stop Server",
                                    command=self.stop_server, state=DISABLED)
        self.stop_btn.pack(side=LEFT, padx=(0, 10))

        self.status_var = StringVar(value="Ready")
        self.vram_var = StringVar(value="")
        ttk.Label(self.btn_frame, textvariable=self.status_var, foreground="gray").pack(side=LEFT, padx=(10, 0))
        ttk.Label(self.btn_frame, textvariable=self.vram_var, foreground="darkgray").pack(side=LEFT, padx=(5, 0))

        # ── Log ──
        log_frame = ttk.LabelFrame(root, text="Log", padding=(4, 4))
        log_frame.pack(fill=BOTH, expand=True, padx=12, pady=(2, 0))

        self.log_text = Text(log_frame, height=6, wrap=WORD, state=DISABLED,
                              font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4",
                              insertbackground="white")
        scrollbar = Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        # ── Restore last selection ──
        last_model = self.config.get("last_model")
        if last_model and last_model in MODELS:
            self.model_var.set(last_model)
            self.on_model_select()
        elif MODELS:
            self.model_var.set(list(MODELS.keys())[0])
            self.on_model_select()

        last_backend = self.config.get("last_backend", "ROCm")
        self.backend_var.set(last_backend)
        self.on_backend_change()

    # ── Event Handlers ───────────────────────────────────────────────
    def on_model_select(self, *args):
        model_name = self.model_var.get()

        # Handle custom model browse
        if model_name == "📂 Browse for model...":
            path = filedialog.askopenfilename(
                title="Select a GGUF model file",
                filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")]
            )
            if not path:
                self.model_var.set(self.config.get("last_model", list(MODELS.keys())[0]))
                return
            path = Path(path)

            # Read GGUF metadata for auto-config
            meta = read_gguf_meta(path)
            default_ctx = meta.get("ctx") or 8192

            name = f"Custom: {path.name}"
            MODELS[name] = {
                "key": "Custom",
                "path": str(path),
                "ctx": default_ctx, "batch": 4096, "ubatch": 1024,
                "speed": "—", "template": None, "jinja": False,
                "modes": ["Default"],
                "mode_overrides": {"Default": {
                    "ctx": default_ctx, "batch": 4096, "ubatch": 1024,
                    "kv_k": "q8_0", "kv_v": "q8_0"
                }},
                "notes": f"Custom model: {path.name} (auto-detected ctx={default_ctx})"
            }
            # Persist custom model
            save_custom_models(MODELS)

            base = [m for m in list(MODELS.keys()) if m != name
                    and not m.startswith("─") and m != "📂 Browse for model..."]
            self.model_combo["values"] = base + ["─" * 30, "📂 Browse for model..."]
            self.model_var.set(name)
            self.on_model_select()
            return

        if model_name.startswith("─"):
            self.model_var.set(self.config.get("last_model", list(MODELS.keys())[0]))
            return

        info = MODELS.get(model_name)
        if not info:
            return
        self.model_note.config(text=info["notes"])
        self.ctx_var.set(info["ctx"])
        self.batch_var.set(info["batch"])
        self.ubatch_var.set(info["ubatch"])
        self.jinja_var.set(info.get("jinja", False))
        tpl_name = info.get("template")
        if tpl_name:
            tpl_path = SCRIPT_DIR / tpl_name
            if tpl_path.exists():
                self.template_var.set(tpl_name)
            else:
                tpl_path2 = DIST_DIR / tpl_name
                if tpl_path2.exists():
                    self.template_var.set(tpl_name)
                else:
                    tpl_path3 = DIST_DIR / "templates" / tpl_name
                    if tpl_path3.exists():
                        self.template_var.set(tpl_name)
                    else:
                        self.template_var.set("None")
        else:
            self.template_var.set("None")
        self.mode_combo["values"] = info.get("modes", ["Default"])
        self.mode_var.set("Default")
        self.override_var.set("")
        self.extra_var.set("")
        self.on_mode_change()
        if info["key"] == "Bonsai":
            self.log(f"Selected: {model_name}  |  {info['speed']}  |  Modes: {' | '.join(info['modes'])}")
        else:
            self.log(f"Selected: {model_name}  |  {info['speed']}")

    def on_backend_change(self, *args):
        backend = self.backend_var.get()
        if backend == "ROCm":
            hip = find_hip_sdk()
            prism_dir = SCRIPT_DIR / "llama.cpp" / "bin" / "prism-hip"
            has_prism = (prism_dir / "llama-server.exe").exists()
            if hip and has_prism:
                self.hip_status.config(text="ROCm: PrismML (prism-hip) — 54 t/s for Bonsai", foreground="green")
            elif hip:
                self.hip_status.config(text="ROCm: mainline only — Bonsai won't work", foreground="orange")
            else:
                self.hip_status.config(text="ROCm: ✗ SDK not found — pip install rocm-sdk", foreground="red")
        elif backend == "ROCm-mainline":
            hip = find_hip_sdk()
            if hip:
                self.hip_status.config(text="ROCm: mainline (no Q2_0 support)", foreground="green")
            else:
                self.hip_status.config(text="ROCm: ✗ SDK not found — pip install rocm-sdk", foreground="red")
        else:
            self.hip_status.config(text="Vulkan runtime is built into the GPU driver", foreground="gray")

    def on_mode_change(self, *args):
        """Update quick settings when mode changes."""
        model_name = self.model_var.get()
        info = MODELS.get(model_name)
        if not info:
            return
        mode = self.mode_var.get()
        overrides = info.get("mode_overrides", {}).get(mode)
        if not overrides:
            return
        self.ctx_var.set(overrides.get("ctx", info["ctx"]))
        self.batch_var.set(overrides.get("batch", info["batch"]))
        self.ubatch_var.set(overrides.get("ubatch", info["ubatch"]))
        self.log(f"  Mode: {mode} — ctx={overrides.get('ctx')}, batch={overrides.get('batch')}, kv={overrides.get('kv_k', 'default')}")

    def _refresh_templates(self):
        """Scan for available .jinja template files."""
        templates = ["None"]
        for d in (SCRIPT_DIR, SCRIPT_DIR / "llama.cpp", SCRIPT_DIR / "llama.cpp" / "templates"):
            if d.is_dir():
                for f in sorted(d.glob("*.jinja")):
                    if f.name not in templates:
                        templates.append(f.name)
        self.template_combo["values"] = templates

    def _on_template_manual(self, *args):
        tpl = self.template_var.get()
        if tpl and tpl != "None":
            self.log(f"  Template: {tpl}")
        else:
            self.log("  Template: None (raw prompt)")

    def toggle_advanced(self):
        if self.show_adv.get():
            self.adv_frame.pack(fill=X, padx=12, pady=(4, 2), before=self.btn_frame)
        else:
            self.adv_frame.pack_forget()

    # ── Logging ──────────────────────────────────────────────────────
    def log(self, msg):
        self.log_text.config(state=NORMAL)
        self.log_text.insert(END, msg + "\n")
        self.log_text.see(END)
        self.log_text.config(state=DISABLED)

    # ── Launch / Stop ────────────────────────────────────────────────
    def launch(self):
        if self.running:
            messagebox.showwarning("Already Running", "Server is already running. Stop it first.")
            return

        model_name = self.model_var.get()
        info = MODELS.get(model_name)
        if not info:
            messagebox.showerror("Error", "No model selected.")
            return

        backend = self.backend_var.get()
        is_bonsai = info["key"] == "Bonsai"
        is_custom = info["key"] == "Custom"

        # Resolve model path
        if is_custom:
            model_path = Path(info["path"])
        else:
            # Auto-switch GGUF variant based on backend for Bonsai
            if is_bonsai and backend == "Vulkan":
                # Vulkan uses g64 format
                model_path = resolve_path(info["path"])  # path already g64
            elif is_bonsai and backend.startswith("ROCm"):
                # ROCm: try Q2_0 (prism-hip) first, fall back to g64
                fallback = info.get("path_fallback")
                p = resolve_path(fallback) if fallback else None
                model_path = resolve_path(info["path"])
                if p and p.exists() and (SCRIPT_DIR / "llama.cpp" / "bin" / "prism-hip" / "llama-server.exe").exists():
                    model_path = p  # Use Q2_0 for PrismML ROCm
            else:
                model_path = resolve_path(info["path"])
                if not model_path.exists():
                    for key in ("path_fallback",):
                        fp = info.get(key)
                        if fp:
                            p = resolve_path(fp)
                            if p.exists():
                                model_path = p
                                break

        if not model_path or not model_path.exists():
            tried = [info["path"]]
            repo = info.get("hf_repo")
            hf_file = info.get("hf_file")
            msg = f"Model file not found.\n\nTried:\n" + "\n".join(tried)
            if repo and hf_file:
                msg += f"\n\nDownload:\nhuggingface-cli download {repo} {hf_file} --local-dir <models_dir>"
                msg += "\n\nOr click Download to use HuggingFace (requires huggingface-cli)"
            msgbox = messagebox.askyesno("Model Not Found",
                                          msg + "\n\nDownload now?",
                                          icon="warning")
            if msgbox and repo and hf_file:
                self._download_model(repo, hf_file, model_path)
                if model_path.exists():
                    self.launch()
                return
            return

        # Pick server binary
        if backend == "ROCm":
            prism_dir = SCRIPT_DIR / "llama.cpp" / "bin" / "prism-hip"
            if (prism_dir / "llama-server.exe").exists():
                server_dir = prism_dir
            else:
                server_dir = DIST_DIR / "bin" / "rocm"
            hip_path = find_hip_sdk()
            if not hip_path:
                messagebox.showerror("ROCm Error",
                                      "ROCm SDK not found.\nInstall with: pip install rocm-sdk\nOr switch to Vulkan backend.")
                return
        elif backend == "ROCm-mainline":
            server_dir = DIST_DIR / "bin" / "rocm"
            hip_path = find_hip_sdk()
            if not hip_path:
                messagebox.showerror("ROCm Error",
                                      "ROCm SDK not found.\nInstall with: pip install rocm-sdk\nOr switch to Vulkan backend.")
                return
        else:
            server_dir = DIST_DIR / "bin" / "vulkan"

        server_exe = server_dir / "llama-server.exe"
        if not server_exe.exists():
            messagebox.showerror("Error", f"llama-server.exe not found at:\n{server_exe}\n\nRun setup.ps1 to download binaries.")
            return

        # Build command
        cmd = [
            str(server_exe),
            "-m", str(model_path),
            "--host", "0.0.0.0",
            "--port", str(self.port_var.get()),
            "-c", str(self.ctx_var.get()),
            "-ngl", str(self.ngl_var.get()),
            "-t", str(self.threads_var.get()),
            "-tb", "16",
            "-b", str(self.batch_var.get()),
            "--ubatch-size", str(self.ubatch_var.get()),
            "--kv-offload",
            "--reasoning", "off",
            "--parallel", "1",
            "-ctxcp", "128",
        ]

        # KV cache type from mode override
        mode = self.mode_var.get()
        mode_overrides = info.get("mode_overrides", {}).get(mode, {})
        kv_k = mode_overrides.get("kv_k", "q8_0")
        kv_v = mode_overrides.get("kv_v", "q8_0")
        cmd.extend(["--cache-type-k", kv_k, "--cache-type-v", kv_v])

        if is_bonsai:
            cmd.extend(["--temp", "0.7", "--top-p", "0.95", "--top-k", "20"])

        if self.nommap_var.get():
            cmd.append("--no-mmap")
        if self.mlock_var.get():
            cmd.append("--mlock")
        if self.flash_var.get():
            cmd.extend(["--flash-attn", "on"])

        # Chat template
        selected_template = self.template_var.get()
        if selected_template and selected_template != "None":
            template_path = SCRIPT_DIR / selected_template
            if not template_path.exists():
                template_path = DIST_DIR / selected_template
            if not template_path.exists():
                template_path = DIST_DIR / "templates" / selected_template
            if self.jinja_var.get():
                cmd.extend(["--jinja"])
            if template_path.exists():
                cmd.extend(["--chat-template-file", str(template_path)])

        # Mode-specific flags
        if "Full" in mode and is_bonsai:
            cmd.extend(["--reasoning", "on"])
        elif "Full" in mode:
            cmd.extend([
                "--override-tensor", r"blk\.*\.ffn_(gate_up|gate|up|down)_exps\.weight=CPU",
                "--no-kv-unified",
                "--cpu-range", "0-15", "--cpu-range-batch", "16-31", "--cpu-strict", "1",
                "--prio-batch", "1", "--poll", "50",
            ])

        if "LongCtx" in mode and not is_bonsai:
            cmd.extend(["--override-tensor", r"blk\.30\.ffn_(gate_up|gate|up|down)_exps\.weight=CPU"])

        if "Thinking" in mode and not ("Full" in mode and is_bonsai):
            cmd.extend(["--reasoning", "on"])

        override = self.override_var.get().strip()
        if override:
            cmd.extend(["--override-tensor", override])

        extra = self.extra_var.get().strip()
        if extra:
            cmd.extend(extra.split())

        # Environment
        env = os.environ.copy()
        env["GGML_CPU_DLL"] = "ggml-cpu-alderlake.dll"
        if backend.startswith("ROCm"):
            env["PATH"] = f"{server_dir};{hip_path}\\bin;{hip_path}\\lib\\llvm\\bin;{env.get('PATH', '')}"

        self.log("─" * 50)
        self.log(f"Launching: {model_name}")
        self.log(f"Backend:  {backend}")
        self.log(f"Port:     {self.port_var.get()}")
        self.log(f"Context:  {self.ctx_var.get()}")
        self.log(f"Command:  {' '.join(cmd)}")

        try:
            self.server_process = subprocess.Popen(
                cmd, env=env, cwd=str(SCRIPT_DIR),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW
            )
            self.running = True
            self.launch_btn.config(state=DISABLED)
            self.stop_btn.config(state=NORMAL)
            port = self.port_var.get()
            self.status_var.set(f"Starting on port {port}...")
            self.root.title(f"● llama.cpp — {model_name.split('(')[0].strip()} :{port}")

            # Start health check thread
            self.health_thread = threading.Thread(target=self._wait_for_health, args=(port,), daemon=True)
            self.health_thread.start()

            # Save config
            self.config["last_model"] = model_name
            self.config["last_backend"] = backend
            save_config(self.config)

            self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
            self.monitor_thread.start()

        except Exception as e:
            self.log(f"ERROR: {e}")
            messagebox.showerror("Launch Failed", str(e))

    def _download_model(self, repo, hf_file, dest):
        """Download a model from HuggingFace in a background thread."""
        self.log(f"Downloading {repo}/{hf_file}...")
        threading.Thread(target=self._do_download, args=(repo, hf_file, dest), daemon=True).start()

    def _do_download(self, repo, hf_file, dest):
        """Download model from HuggingFace. Uses huggingface-cli if available, else raw URL."""
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Try huggingface-cli first
        try:
            result = subprocess.run(
                ["huggingface-cli", "download", repo, hf_file, "--local-dir", str(dest.parent)],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                self.root.after(0, lambda: self.log(f"Downloaded {hf_file} successfully"))
                return
        except FileNotFoundError:
            pass

        # Fallback: raw URL download
        self.root.after(0, lambda: self.log("Trying raw download..."))
        url = f"https://huggingface.co/{repo}/resolve/main/{hf_file}"
        try:
            import urllib.request
            with urllib.request.urlopen(url) as r:
                total = int(r.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 8192
                with open(dest, "wb") as f:
                    while True:
                        chunk = r.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            self.root.after(0, lambda p=pct: self.log(f"  Downloading... {p}%"))
            self.root.after(0, lambda: self.log(f"Downloaded {hf_file} ({dest})"))
            return
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Download failed: {e}"))

        # If we get here, both methods failed
        self.root.after(0, lambda: self.log(
            "Install huggingface-cli: pip install huggingface-hub"))

    def _wait_for_health(self, port):
        """Poll server health endpoint until ready, then update status."""
        import time
        for attempt in range(30):
            time.sleep(0.5)
            if check_server_health(port):
                self.root.after(0, lambda p=port: self.status_var.set(f"● Running on port {p}"))
                self.root.after(0, lambda: self.vram_var.set(""))
                return
        # Even if health check fails, show running
        self.root.after(0, lambda p=port: self.status_var.set(f"Running (health unknown) on port {p}"))

    def _monitor(self):
        try:
            for line in iter(self.server_process.stdout.readline, ""):
                if not line:
                    break
                line = line.rstrip()
                if "CUDA Graph id" in line or "reused" in line:
                    continue
                if line.strip():
                    self.root.after(0, self._log_line, line)
        except Exception:
            pass
        finally:
            self.root.after(0, self._server_stopped)

    def _log_line(self, line):
        self.log(line)

    def _server_stopped(self):
        self.running = False
        self.launch_btn.config(state=NORMAL)
        self.stop_btn.config(state=DISABLED)
        self.status_var.set("Stopped")
        self.root.title("llama.cpp Launcher")

    def stop_server(self):
        if not self.server_process:
            return
        self.log("Stopping server...")
        try:
            self.server_process.terminate()
            self.server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.server_process.kill()
        except Exception as e:
            self.log(f"Error stopping: {e}")
        self._server_stopped()

    def on_close(self):
        if self.running:
            self.stop_server()
        self.root.destroy()


# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────
def main():
    root = Tk()
    app = LauncherApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
