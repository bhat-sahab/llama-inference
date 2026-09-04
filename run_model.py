#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_model.py - launcher & tester for llama-server (replaces run_model.bat)

Same menu flow as the .bat (model -> backend -> vision -> context -> MTP),
plus:
  * CLI overrides so runs/tests can be scripted
  * `test` mode: boots the server, waits for readiness, runs a fixed prompt
    and reports MEASURED pp/tg next to the expected speeds from the table
  * `list`: shows all models/backends with on-disk status
  * GPU reset (PnP) + ADLX power profiles, same as the bat

Usage:
  python run_model.py                     # interactive menu (like the bat)
  python run_model.py list
  python run_model.py run   --model ridge --backend vulkan --ctx 32768
  python run_model.py test  --model ridge --backend vulkan --ctx 32768 --tokens 64
  python run_model.py gpu-reset
  python run_model.py power --profile low|stock|max
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
IS_WIN = os.name == "nt"
ROOT = Path(__file__).resolve().parent
MODELS_DIR = (Path(r"C:\Users\BhatSahab\.lmstudio\models") if IS_WIN else
              ROOT / "models")  # symlink in the repo -> external LM Studio models folder
BACKENDS = ROOT / "backends" / "bin"
HOST = "0.0.0.0"
DEFAULT_PORT = 8081

SAMPLING = ["--temp", "1.0", "--top-p", "0.95", "--top-k", "20",
            "--min-p", "0.0", "--presence-penalty", "0.0", "--repeat-penalty", "1.0"]
REASONING = ["--reasoning", "on", "--reasoning-preserve"]
# env var (JSON) - batch-safe way to pass chat-template kwargs (flag rejected b10181/rocm)
CHAT_TEMPLATE_KWARGS = '{"preserve_thinking": true, "reasoning_effort": "medium"}'
# DFlash2 speculative-decoding draft for Qwen3.8-27B (draft-dflash, added in b10701)
DFLASH_DRAFT = MODELS_DIR / "z-lab" / "Qwen3.8-27B-DFlash2-GGUF" / "Qwen3.8-27B-DFlash2-Q4_K_M.gguf"

# ── Console colors ────────────────────────────────────────────────────────
_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
_B, _E = "\001b[1m", "\001b[0m"
_G = "\001b[32m"  # green
_Y = "\001b[33m"  # yellow
_R = "\001b[31m"  # red
_C = "\001b[36m"  # cyan


def _setup_console() -> None:
    # Win10+ consoles: Python enables VT/ANSI automatically when stdout is a
    # console handle; just make sure text streams are UTF-8 (cmd defaults cp1252)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def p(*args) -> None:
    print(*args)


def b(s) -> str:
    return f"{_B}{s}{_E}" if _COLOR else s


def g(s) -> str:
    return f"{_G}{s}{_E}" if _COLOR else s


def y(s) -> str:
    return f"{_Y}{s}{_E}" if _COLOR else s


def r(s) -> str:
    return f"{_R}{s}{_E}" if _COLOR else s


def c(s) -> str:
    return f"{_C}{s}{_E}" if _COLOR else s


def cls() -> None:
    os.system("cls" if IS_WIN else "clear")


# ── Backend definitions ───────────────────────────────────────────────────
@dataclass
class Backend:
    key: str
    label: str
    dir: Path
    family: str  # "vk" | "rocm"

    @property
    def exe(self) -> Path:
        return self.dir / ("llama-server.exe" if IS_WIN else "llama-server")


if IS_WIN:
    MOE_BACKENDS = [
        Backend("vulkan",    "Vulkan b10181 (MoE tg king ~46)",        BACKENDS / "vulkan-b10181",  "vk"),
        Backend("selfbuilt", "ROCm selfbuilt (b10488+fix, MTP +31%)",  BACKENDS / "rocm-selfbuilt", "rocm"),
        Backend("stew",      "ROCm stew675 (b10649, tg ~43)",          BACKENDS / "rocm-stew675",    "rocm"),
        Backend("vpatch",    "Vulkan patched (b10488+pi, exp)",        BACKENDS / "vulkan-patched", "vk"),
        Backend("rocm72",    "ROCm 7.2 selfbuilt (pp king ~3013, exp)", BACKENDS / "rocm-72",       "rocm"),
    ]
    DENSE_BACKENDS = [
        Backend("vulkan",    "Vulkan (b10639, tg ~31-37)",             BACKENDS / "vulkan",          "vk"),
        Backend("stew",      "ROCm stew675 (b10649, pp ~1000, tg ~30)", BACKENDS / "rocm-stew675",    "rocm"),
        Backend("selfbuilt", "ROCm selfbuilt (pp ~900, MTP works)",   BACKENDS / "rocm-selfbuilt",  "rocm"),
    ]
else:
    # Linux: perf fork + MoE expert cache is the MoE winner; Vulkan wins decode on dense.
    MOE_BACKENDS = [
        Backend("moe",       "perf fork + MoE cache (q36 tg ~45 / KAT ~62)", BACKENDS / "rocm-moe-linux",       "moe"),
        Backend("selfbuilt", "ROCm official (q36 pp ~1320, tg ~37)",         BACKENDS / "rocm-linux",           "rocm"),
        Backend("stew",      "ROCm stew675 rdna-boosts (q36 tg ~37)",        BACKENDS / "rocm-stew675-linux",   "rocm"),
        Backend("vulkan",    "Vulkan (q36 pp 513, tg ~34)",                  BACKENDS / "vulkan-linux",         "vk"),
    ]
    DENSE_BACKENDS = [
        Backend("vulkan",    "Vulkan (pp ~1130, tg ~39)",             BACKENDS / "vulkan-linux",       "vk"),
        Backend("stew",      "ROCm stew675 (pp ~1130, tg ~28)",       BACKENDS / "rocm-stew675-linux", "rocm"),
        Backend("selfbuilt", "ROCm official (pp ~1085, tg ~27)",      BACKENDS / "rocm-linux",         "rocm"),
    ]


def backends_for(m: "Model") -> list[Backend]:
    return DENSE_BACKENDS if m.is_dense else MOE_BACKENDS


# ── Model definitions ─────────────────────────────────────────────────────
@dataclass
class Model:
    key: str
    label: str
    path: Path
    is_dense: bool
    mtp: bool
    mmproj: Path | None
    kv: tuple[str, str]
    reasoning: bool
    speeds: dict  # ctx -> (pp_str, tg_str)
    cfg: dict     # family -> {ctx: (batch, ubatch, n_cpu_moe)}


MODELS: list[Model] = [
    Model(
        "q36", "Qwen3.6 IQ4_XS MTP (18.2 GB, MTP, vision)",
        MODELS_DIR / "unsloth" / "Qwen3.6-35B-A3B-MTP-GGUF" / "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
        is_dense=False, mtp=True,
        mmproj=MODELS_DIR / "unsloth" / "Qwen3.6-35B-A3B-MTP-GGUF" / "mmproj-F32.gguf",
        kv=("q4_0", "q4_0"), reasoning=False,
        speeds={32768: ("~2900", "~37"), 65536: ("~2500", "~48"),
                131072: ("~2400", "~47"), 262144: ("~1200", "~45")},
        cfg={"vk":   {32768: (4096, 2048, 16), 65536: (4096, 2048, 16),
                      131072: (4096, 2048, 18), 262144: (512, 512, 21)},
             "rocm": {32768: (4096, 4096, 18), 65536: (4096, 4096, 18),
                      131072: (2048, 2048, 20), 262144: (512, 512, 20)},
             "moe":  {32768: (4096, 4096, 18), 65536: (4096, 4096, 18),
                      131072: (2048, 2048, 20), 262144: (512, 512, 21)}},
    ),
    Model(
        "kat", "KAT-Coder IQ4_XS MTP (19.7 GB, vision)",
        MODELS_DIR / "thread13" / "Kwaipilot_KAT-Coder-V2.5-Dev-GGUF-MTP" / "Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.bartowski.mtp.gguf",
        is_dense=False, mtp=True,
        mmproj=None,  # bat has no mmproj path for KAT
        kv=("q4_0", "q4_0"), reasoning=False,
        speeds={32768: ("~2900", "~37"), 65536: ("~2500", "~48"),
                131072: ("~2400", "~47"), 262144: ("~1200", "~45")},
        cfg={"vk":   {32768: (4096, 1024, 14), 65536: (4096, 1024, 14),
                      131072: (2048, 1024, 16), 262144: (512, 512, 18)},
             "rocm": {32768: (4096, 1024, 10), 65536: (4096, 1024, 11),
                      131072: (4096, 1024, 13), 262144: (512, 512, 17)},
             "moe":  {32768: (4096, 1024, 18), 65536: (4096, 1024, 18),
                      131072: (2048, 1024, 20), 262144: (512, 512, 21)}},
    ),
    Model(
        "q38", "Qwen3.8-27B IQ4_XS (14.3 GB, dense, vision)",
        MODELS_DIR / "unsloth" / "Qwen3.8-27B-GGUF" / "Qwen3.8-27B-UD-IQ4_XS.gguf",
        is_dense=True, mtp=True,
        mmproj=MODELS_DIR / "unsloth" / "Qwen3.8-27B-GGUF" / "mmproj-F16.gguf",
        kv=("q4_0", "q4_0"), reasoning=True,
        speeds={32768: ("~589", "~28"), 65536: ("~589", "~28"),
                131072: ("~589", "~28"), 262144: ("❌", "❌")},
        cfg={"vk":   {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                      131072: (512, 512, 0), 262144: (512, 512, 0)},
             "rocm": {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                      131072: (512, 512, 0), 262144: (512, 512, 0)}},
    ),
    Model(
        "q38m", "Qwen3.8-27B Q3_K_M (13.8 GB, dense, vision)",
        MODELS_DIR / "unsloth" / "Qwen3.8-27B-GGUF" / "Qwen3.8-27B-Q3_K_M.gguf",
        is_dense=True, mtp=True,
        mmproj=MODELS_DIR / "unsloth" / "Qwen3.8-27B-GGUF" / "mmproj-F16.gguf",
        kv=("q4_0", "q4_0"), reasoning=True,
        speeds={32768: ("~655", "~37"), 65536: ("~646", "~32"),
                131072: ("~670", "~31"), 262144: ("❌102", "❌")},  # 256K broken (VRAM compute buffers)
        cfg={"vk":   {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                      131072: (512, 512, 0), 262144: (512, 512, 0)},
             "rocm": {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                      131072: (512, 512, 0), 262144: (512, 512, 0)}},
    ),
    Model(
        "q8", "Qwen3.8-9B-Distill Q8_0 (9.1 GB, q8 KV)",
        MODELS_DIR / "empero-ai" / "Qwen3.8-9B-Distill-GGUF" / "Qwen3.8-9B-Q8_0.gguf",
        is_dense=True, mtp=True,
        mmproj=None,  # no 9B-compatible mmproj on disk (27B ones die - hidden-dim mismatch)
        kv=("q8_0", "q8_0"), reasoning=True,
        speeds={32768: ("~2670", "~58"), 65536: ("~2670", "~58"),
                131072: ("~2670", "~58"), 262144: ("~2670", "~58")},
        cfg={"vk":   {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                      131072: (512, 512, 0), 262144: (512, 512, 0)},
             "rocm": {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                      131072: (512, 512, 0), 262144: (512, 512, 0)}},
    ),
    Model(
        "ridge", "Qwen3.8-27B Ridge-3.7bpw (12.6 GB, dense, ALL ctx, vision)",
        MODELS_DIR / "empero-ai" / "Qwen3.8-27B-Ridge-GGUF" / "Qwen3.8-27B-Ridge-3.7bpw.gguf",
        is_dense=True, mtp=True,
        mmproj=MODELS_DIR / "empero-ai" / "Qwen3.8-27B-Ridge-GGUF" / "mmproj-Qwen3.8-27B-BF16.gguf",
        kv=("q4_0", "q4_0"), reasoning=True,
        speeds={32768: ("~722", "~34"), 65536: ("~770", "~35"),
                131072: ("~733", "~29"), 262144: ("~720", "~27")},
        cfg={"vk":   {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                      131072: (1024, 2048, 0), 262144: (512, 512, 0)},
             "rocm": {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                      131072: (1024, 2048, 0), 262144: (512, 512, 0)}},
    ),
    Model(
        "q2x", "Qwen3.8-27B Q2_K_XL (9.2 GB, dense, vision)",
        MODELS_DIR / "unsloth" / "Qwen3.8-27B-GGUF" / "Qwen3.8-27B-UD-Q2_K_XL.gguf",
        is_dense=True, mtp=True,
        mmproj=MODELS_DIR / "unsloth" / "Qwen3.8-27B-GGUF" / "mmproj-F16.gguf",
        kv=("q4_0", "q4_0"), reasoning=True,
        speeds={32768: ("~650", "~45"), 65536: ("~650", "~45"),
                131072: ("~650", "~45"), 262144: ("~?", "~?")},
        cfg={"vk":   {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                      131072: (512, 512, 0), 262144: (512, 512, 0)},
             "rocm": {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                      131072: (512, 512, 0), 262144: (512, 512, 0)}},
    ),
]

MODEL_BY_KEY = {m.key: m for m in MODELS}
CTX_CHOICES = [32768, 65536, 131072, 262144]


def find_model(key: str) -> Model:
    m = MODEL_BY_KEY.get(key)
    if not m:
        raise SystemExit(r(f"Unknown model '{key}'. Known: {', '.join(MODEL_BY_KEY)}"))
    return m


# ── MTP flags (per model / backend family) ────────────────────────────────
def normal_mtp_flags(m: Model) -> list[str]:
    flags = ["--spec-type", "draft-mtp"]
    if m.key in ("q38", "q38m"):
        # 1-layer MTP: n_max MUST be 1 (default 3 = CRASH, 2 = tg collapse)
        flags += ["--spec-draft-n-max", "1", "--spec-draft-p-min", "0.82"]
    elif m.key == "q2x":
        flags += ["--spec-draft-n-max", "2", "--spec-draft-p-min", "0.82"]
    return flags + ["--cache-type-k-draft", "q4_0", "--cache-type-v-draft", "q4_0"]


def mtp_flags(m: Model, fam: str, backend_key: str, mode: str = "tuned") -> list[str]:
    # perf fork (moe): MTP is VRAM-tight on top of the expert cache (112 slots)
    if backend_key == "moe" and m.key in ("q36", "kat"):
        n_max = "3" if m.key == "q36" else "1"   # q36 caps at 3; KAT only fits 1
        return ["--spec-type", "draft-mtp", "--spec-draft-n-max", n_max,
                "--cache-type-k-draft", "q4_0", "--cache-type-v-draft", "q4_0"]
    if mode == "normal":
        return normal_mtp_flags(m)
    if m.key == "q36":
        # Only works on rocm-selfbuilt (+31% tg); others: no spec support.
        return ["--spec-type", "draft-mtp"]
    if m.key in ("q38", "q38m", "q2x"):
        return normal_mtp_flags(m)
    if m.key == "ridge":
        if fam == "vk":
            # combined draft-mtp,ngram-mod (eaman cfg): +30% tg dense
            return ["--spec-type", "draft-mtp,ngram-mod", "--spec-draft-p-min", "0.82",
                    "--spec-draft-n-max", "2", "--cache-type-k-draft", "q4_0",
                    "--cache-type-v-draft", "q4_0", "--spec-ngram-mod-n-match", "24",
                    "--spec-ngram-mod-n-min", "8", "--spec-ngram-mod-n-max", "32"]
        # adaptive MTP (stew rdna-boosts): +55% tg on ROCm
        return ["--spec-type", "draft-mtp-adaptive",
                "--spec-draft-n-min-adaptive", "3", "--spec-draft-n-max", "8"]
    return ["--spec-type", "draft-mtp"]  # q8


def mtp_allowed(m: Model, backend: Backend) -> bool:
    if not m.mtp:
        return False
    if m.key in ("q36", "kat") and backend.key == "stew":
        return False  # stew fork MTP crashes qwen35moe (256-expert MoE)
    if m.key == "q36" and backend.key not in ("selfbuilt", "moe"):
        return False  # vulkan-b10181 no spec support; moe/selfbuilt only
    return True


# ── DFlash2 flags (draft-dflash speculative decoding, b10701+) ─────────────
_VER_CACHE: dict[str, int | None] = {}


def backend_build(be: Backend) -> int | None:
    """Numeric build from `llama-server --version` (e.g. 10701), cached per exe."""
    key = str(be.exe)
    if key not in _VER_CACHE:
        try:
            # version banner goes to stderr on llama-server
            out = subprocess.run([str(be.exe), "--version"], capture_output=True,
                                 text=True, timeout=10)
            text = (out.stdout or "") + (out.stderr or "")
            m = re.search(r"\(build\s*(\d+)", text)
            _VER_CACHE[key] = int(m.group(1)) if m else None
        except Exception:
            _VER_CACHE[key] = None
    return _VER_CACHE[key]


def dflash_flags() -> list[str]:
    # KV4 draft cache + n-max 4 (measured sweet spot: +108% tg on Q2_K_XL/ROCm).
    return ["--spec-type", "draft-dflash", "-md", str(DFLASH_DRAFT),
            "--spec-draft-n-max", "4",
            "--cache-type-k-draft", "q4_0", "--cache-type-v-draft", "q4_0"]


def dflash_allowed(m: Model, be: Backend) -> bool:
    # All Qwen3.8-27B targets. The draft costs ~2 GiB of compute buffers, so it
    # may not fit alongside the largest quant on a 16 GiB GPU - user picks what
    # fits. (q8 / 9B distill is excluded: a 27B draft would slow it down.)
    if m.key not in ("q38", "q38m", "q2x", "ridge"):
        return False
    if not DFLASH_DRAFT.exists():
        return False
    ver = backend_build(be)
    return ver is not None and ver >= 10701


# ── Config resolution ─────────────────────────────────────────────────────
@dataclass
class RunConfig:
    model: Model
    backend: Backend
    ctx: int
    batch: int
    ubatch: int
    ncpu_moe: int
    kv: list[str]
    cache_ram: int
    extra: list[str]
    spec: list[str]
    mmproj: list[str]
    port: int


def resolve(m: Model, b: Backend, ctx: int, use_mtp: bool,
            use_vision: bool, port: int, mtp_mode: str = "tuned",
            use_dflash: bool = False) -> RunConfig:
    if mtp_mode not in ("normal", "tuned"):
        raise ValueError(f"Unknown MTP mode: {mtp_mode}")
    table = m.cfg[b.family]
    batch, ubatch, ncmoe = table[ctx]
    kv = list(m.kv)
    cache_ram = 0
    extra: list[str] = []

    # Honor backend support even for CLI callers.
    mtp_on = use_mtp and mtp_allowed(m, b)
    dflash_on = use_dflash and dflash_allowed(m, b)

    # Ridge's tuned Vulkan preset uses a different batching and KV configuration.
    if m.key == "ridge" and b.family == "vk" and mtp_on and mtp_mode == "tuned":
        batch, ubatch, ncmoe = 1024, 128, 0
        kv = ["q4_1", "q4_1"]
        cache_ram = 6000
        extra = ["--fit-target", "30", "--no-warmup", "--ctx-checkpoints", "96"]

    if dflash_on:
        # DFlash2 draft adds ~2 GiB of compute buffers -> force small batches
        # (2048 OOMs on the 16 GiB RX 9070 XT with the draft loaded).
        batch, ubatch, ncmoe = 512, 512, 0

    spec = dflash_flags() if dflash_on else (mtp_flags(m, b.family, b.key, mtp_mode) if mtp_on else [])

    # perf fork: MoE expert cache - hot routed experts kept in VRAM (112 slots fits MTP)
    if b.key == "moe" and m.key in ("q36", "kat"):
        trace = ROOT / "traces" / f"{m.key}-routing.csv"
        if trace.exists():
            extra += ["--moe-cache-profile", str(trace), "--moe-cache-slots", "112"]

    mmproj = ["--mmproj", str(m.mmproj)] if (
        use_vision and m.mmproj and b.key != "stew"  # stew fork: MUL_MAT crash on vision encoder
    ) else []
    return RunConfig(m, b, ctx, batch, ubatch, ncmoe, kv, cache_ram, extra, spec, mmproj, port)


def build_cmd(rc: RunConfig) -> list[str]:
    return [str(rc.backend.exe),
            "-m", str(rc.model.path),
            "-c", str(rc.ctx),
            "-t", "8",
            "-b", str(rc.batch),
            "--ubatch-size", str(rc.ubatch),
            "-ctk", rc.kv[0], "-ctv", rc.kv[1],
            "--flash-attn", "on",
            "--load-mode", "none",
            "--n-cpu-moe", str(rc.ncpu_moe),
            "--cache-ram", str(rc.cache_ram),
            "--no-repack", "--fit", "off",
            "-np", "1",
            "--host", HOST, "--port", str(rc.port),
            "--jinja"] + rc.spec + rc.mmproj + SAMPLING + \
        (REASONING if rc.model.reasoning else []) + rc.extra


def launch_env(rc: RunConfig) -> dict:
    env = os.environ.copy()
    if rc.model.reasoning:
        env["LLAMA_ARG_CHAT_TEMPLATE_KWARGS"] = CHAT_TEMPLATE_KWARGS
    return env


# ── Helpers ───────────────────────────────────────────────────────────────
def kill_port(port: int) -> None:
    """Kill any stale process holding the port (fixes 'Server stopped (code 1)')."""
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    except Exception:
        return
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[1] == "LISTENING":
            m = re.match(r":?(\d+)$", parts[3])
            if m and int(m.group(1)) == port:
                pid = parts[4]
                subprocess.run(["taskkill", "/f", "/pid", pid],
                               capture_output=True)


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def gpu_reset() -> None:
    ps1 = ROOT / "tools" / "reset-gpu.ps1"
    if not ps1.exists():
        p(r(f"reset-gpu.ps1 not found: {ps1}"))
        return
    p(b("GPU Reset - RX 9070 XT (PnP cycle)"), " ")
    p("  Screen will go black for a few seconds.")
    p("  Fixes GPU stuck at idle clocks (tg collapse).")
    p()
    if is_admin():
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                        "-File", str(ps1)])
    else:
        p(y("  Not elevated - requesting UAC elevation..."))
        fmt = str(ps1).replace("{", "{{").replace("}", "}}")
        cmd = ('Start-Process powershell -Verb RunAs -Wait -ArgumentList '
               '@("-NoProfile","-ExecutionPolicy","Bypass","-File","{0}")'
               .format(fmt))
        subprocess.run(["powershell", "-NoProfile", "-Command", cmd])
    p(g("  GPU reset done."))


def power_profile(profile: str) -> None:
    exe = ROOT / "tools" / "adlx-profile.exe"
    if not exe.exists():
        p(r(f"adlx-profile.exe not found: {exe}"))
        return
    args = {
        "low":   ["apply", "--power-limit", "-30"],
        "stock": ["reset"],
        "max":   ["apply", "--power-limit", "10"],
    }
    if profile not in args:
        raise SystemExit(f"Bad profile '{profile}' (low|stock|max)")
    p(b(f"GPU Power Profile: {profile}"), " ")
    subprocess.run([str(exe)] + args[profile])
    p(g("  Done."))


def summarize(rc: RunConfig, dry: bool = False) -> None:
    p(b("=============================================="))
    p(b("  Starting llama-server" + (" (dry-run)" if dry else "...")))
    p(b("=============================================="))
    p(f"  Model   : {rc.model.path}")
    p(f"  Backend : {rc.backend.exe}")
    p(f"  Context : {rc.ctx}   Batch: {rc.batch}   UBatch: {rc.ubatch}   "
      f"ncpu-moe: {rc.ncpu_moe}   KV: {rc.kv[0]}/{rc.kv[1]}")
    p(f"  Spec    : {' '.join(rc.spec)}" if rc.spec else "  Spec    : off")
    p(f"  Vision  : {' '.join(rc.mmproj)}" if rc.mmproj else "  Vision  : off")
    p()
    p(f"  Server: http://localhost:{rc.port}")
    if not dry:
        p("  Press Ctrl+C to stop")
    p(b("=============================================="))
    p()
    if dry:
        p(c("Command line:"))
        p(" ".join(f'"{x}"' if " " in x else x for x in build_cmd(rc)))
        p()


# ── Server run ────────────────────────────────────────────────────────────
def run_server(rc: RunConfig) -> None:
    kill_port(rc.port)
    summarize(rc)
    cmd = build_cmd(rc)
    proc = subprocess.Popen(cmd, env=launch_env(rc))
    try:
        rc.code = proc.wait()  # type: ignore[attr-defined]
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        rc.code = 130  # type: ignore[attr-defined]
    p()
    p(f"Server stopped (code {rc.code}). Returning to main menu...")


# ── Test mode ─────────────────────────────────────────────────────────────
def http_json(url: str, payload: dict | None = None, timeout: float = 30.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if payload else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_ready(port: int, proc: subprocess.Popen, load_timeout: float, log: Path) -> bool:
    p(f"  Waiting for server (timeout {load_timeout:.0f}s)...")
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            p(r(f"  Server exited early (code {proc.returncode})."))
            p("  --- last log lines ---")
            try:
                lines = log.read_text(errors="replace").splitlines()
                for ln in lines[-25:]:
                    p("  " + ln)
            except Exception:
                pass
            return False
        try:
            urllib.request.urlopen(f"http://localhost:{port}/", timeout=2).read()
            return True
        except Exception:
            pass
        if time.time() - t0 > load_timeout:
            p(r("  Server never became ready."))
            return False
        time.sleep(1.0)


def run_test(rc: RunConfig, prompt: str, n_tokens: int, load_timeout: float) -> None:
    kill_port(rc.port)
    summarize(rc)
    log = ROOT / "server_test.log"
    cmd = build_cmd(rc)
    lf = open(log, "wb")
    p(f"  Server log: {log}")
    p(f"  Test: {n_tokens} tokens, prompt: {prompt!r}")
    proc = subprocess.Popen(cmd, env=launch_env(rc), stdout=lf, stderr=subprocess.STDOUT)
    try:
        ok = wait_ready(rc.port, proc, load_timeout, log)
        if not ok:
            return
        p(g("  Server ready. Running prompt test..."))
        res = http_json(f"http://localhost:{rc.port}/completion",
                        {"prompt": prompt, "n_tokens": n_tokens,
                         "stream": False, "temperature": 0.0},
                        timeout=300)
        t = res.get("timings", {})
        pp_tok, pp_t = t.get("tokens_prompt"), t.get("t_prompt_process")
        tg_tok, tg_t = t.get("tokens_predicted"), t.get("t_predict")
        pp = pp_tok / pp_t if pp_tok and pp_t else None
        tg = tg_tok / tg_t if tg_tok and tg_t else None
        p()
        p(b("  ── Measured ─────────────────────────────"))
        p(f"  prompt : {pp_t:.2f} s for {pp_tok} tok -> {pp:6.1f} t/s" if pp else "  prompt : n/a")
        p(f"  decode : {tg_t:.2f} s for {tg_tok} tok -> {tg:6.1f} t/s" if tg else "  decode : n/a")
        exp_pp, exp_tg = rc.model.speeds.get(rc.ctx, ("?", "?"))
        p(b(f"  ── Expected {rc.ctx} ──────────────────────"))
        p(f"  pp {exp_pp} t/s | tg {exp_tg} t/s  (launcher table, Vulkan clean state)")
        p(b("  ── Output ───────────────────────────────"))
        print("  " + (res.get("content") or res.get("content_str") or "").strip())
    finally:
        p()
        p("  Stopping server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        lf.close()
    p(g("  Test done."))


# ── Interactive menu ──────────────────────────────────────────────────────
def _pick(options: list[tuple[int, str]], prompt: str) -> int | None:
    while True:
        try:
            s = input(f"{prompt} ").strip()
        except (EOFError, KeyboardInterrupt):
            p()
            raise SystemExit(0)
        if s.isdigit():
            if 1 <= int(s) <= len(options):
                return int(s)
        p(r(f"  Enter a number 1-{len(options)}"))


def main_menu() -> None:
    # loop like the bat: after run/test/gpu/power actions, return to the menu
    while True:
        if _main_menu_once():
            break


def _main_menu_once() -> bool:
    cls()
    p(b("=============================================="))
    p(b("  Qwen3.6 / KAT-Coder / Qwen3.8 - Launcher (py)"))
    p(b("=============================================="))
    p()
    opts = []
    for i, m in enumerate(MODELS, 1):
        mark = g("✅") if m.path.exists() else r("❌ MISSING")
        opts.append((i, f"[{i}] {m.label}   {mark}"))
    quit_opt = len(MODELS) + 1
    if IS_WIN:
        opts.append((quit_opt, f"[{quit_opt}] Reset GPU (PnP cycle - fixes stuck idle clocks)"))
        quit_opt += 1
        opts.append((quit_opt, f"[{quit_opt}] GPU Power Profile (ADLX: low/stock/max)"))
        quit_opt += 1
    opts.append((quit_opt, f"[{quit_opt}] Quit"))
    for _, o in opts:
        p(o)
    p()
    sel = _pick(opts, b("  Choice: "))
    if sel is None:
        return
    if IS_WIN:
        if sel == len(MODELS) + 1:
            gpu_reset()
            return
        if sel == len(MODELS) + 2:
            prof = _pick([(1, "[1] Low power (-30%, quiet)"), (2, "[2] Stock"),
                          (3, "[3] Max power (+10%)")], b("  Profile: "))
            power_profile(["low", "stock", "max"][prof - 1])
            return
    if sel == quit_opt:
        return True  # quit -> exit loop

    m = MODELS[sel - 1]
    if not m.path.exists():
        p(r(f"  Model file missing: {m.path}"))
        return

    # ── backend ──
    cls()
    p(b(f"Model: {m.path}"))
    p(b("=============================================="))
    p()
    p(b("Select backend:"))
    opts = []
    for i, be in enumerate(backends_for(m), 1):
        mark = g("✅") if be.exe.exists() else r("❌ MISSING")
        opts.append((i, f"  [{i}] {be.label}  {mark}"))
    opts.append((len(backends_for(m)) + 1, f"  [{len(backends_for(m)) + 1}] Back"))
    sel = _pick(opts, b("  Backend: "))
    if sel is None or sel > len(backends_for(m)):
        return
    be = backends_for(m)[sel - 1]

    # ── vision ──
    use_vision = False
    if m.mmproj and be.key != "stew":
        cls()
        p(b(f"Model  : {m.path}"))
        p(b(f"Backend: {be.exe}"))
        p(b("=============================================="))
        p()
        if not m.mmproj.exists():
            p(r(f"  mmproj not on disk: {m.mmproj}"))
            p()
        else:
            sel = _pick([(1, "  [1] Vision ON  (mmproj loaded, images via /v1/chat/completions)"),
                         (2, "  [2] Vision OFF")], b("  Vision: "))
            if sel == 1 and m.mmproj.exists():
                use_vision = True
    elif be.key == "stew" and m.mmproj:
        cls()
        p(b("Vision NOT available on stew backend"), " ")
        p("(b10486 fork: MUL_MAT crash on vision encoder)")
        p(b("Continuing without vision..."))
        p()
        time.sleep(1.5)

    # ── context ──
    cls()
    p(b(f"Model  : {m.path}"))
    p(b(f"Backend: {be.exe}"))
    p(b("=============================================="))
    p()
    p(b("Select context window (pp | tg t/s):"))
    opts = []
    for i, ctx in enumerate(CTX_CHOICES, 1):
        pp, tg = m.speeds.get(ctx, ("?", "?"))
        label = "K" if ctx >= 1024 else "K"
        opts.append((i, f"  [{i}] {ctx // 1024}K   pp {pp} | tg {tg}"))
    sel = _pick(opts, b("  Context: "))
    if sel is None:
        return
    ctx = CTX_CHOICES[sel - 1]
    if m.key == "q38m" and ctx == 262144:
        p(y("  WARNING: Q3_K_M 256K is broken (VRAM compute buffers, pp ~100)"))

    # ── MTP / DFlash2 ──
    use_mtp = False
    use_dflash = False
    mtp_mode = "tuned"
    dflash_ok = dflash_allowed(m, be)
    if mtp_allowed(m, be) or dflash_ok:
        cls()
        p(b(f"Model  : {m.path}"))
        p(b(f"Backend: {be.exe}"))
        p(b(f"Context: {ctx}"))
        p(b("=============================================="))
        p()
        note = {
            "q36": "(Qwen3.6 +38% on selfbuilt)",
            "ridge": "(Ridge vulkan MTP+ngram +30% / adaptive +55% rocm)",
        }.get(m.key, "")
        opts = [(1, "  [1] Normal MTP  (draft-mtp)"),
                (2, f"  [2] Tuned MTP   {note}"),
                (3, "  [3] MTP OFF     (safe)")]
        if dflash_ok:
            opts.append((4, "  [4] DFlash2     (draft-dflash, KV4 draft, ~2x tg)"))
        sel = _pick(opts, b("  MTP: "))
        if sel == 4:
            use_dflash = True
        else:
            mtp_mode = "normal" if sel == 1 else "tuned"
            use_mtp = sel in (1, 2)
    elif m.mtp:
        p(y(f"  MTP not supported on {be.key} for {m.key} - continuing without MTP."))

    # ── mode ──
    cls()
    p(b(f"Model  : {m.path}"))
    p(b(f"Backend: {be.exe}"))
    spec_txt = "DFlash2" if use_dflash else ("MTP" if use_mtp else "off")
    p(b(f"Context: {ctx}   Spec: {spec_txt}   "
        f"Vision: {'on' if use_vision else 'off'}"))
    p(b("=============================================="))
    p()
    sel = _pick([(1, "  [1] RUN    - start the server (Ctrl+C to stop)"),
                 (2, "  [2] TEST   - quick bench: boot, measure pp/tg, stop"),
                 (3, "  [3] TEST   - like above with custom prompt"),
                 (4, "  [4] Back")], b("  Mode: "))
    if sel == 4:
        return

    rc = resolve(m, be, ctx, use_mtp, use_vision, DEFAULT_PORT, mtp_mode, use_dflash)

    if sel == 1:
        run_server(rc)
        return
    if sel in (2, 3):
        prompt = "Count from 1 to 100."
        if sel == 3:
            prompt = input("  Prompt: ").strip() or prompt
        run_test(rc, prompt, 64, load_timeout=300.0)
        return


# ── list ──────────────────────────────────────────────────────────────────
def list_all() -> None:
    p(b("Models:"))
    for m in MODELS:
        ok = m.path.exists()
        size = f" ({m.path.stat().st_size / 2**30:.1f} GB)" if ok else ""
        p(f"  [{'✅' if ok else '❌'}] {m.key:<6} {m.label}{size}")
        if not ok:
            p(f"        {r(str(m.path))}")
    p()
    p(b("Backends:"))
    for grp, name in ((MOE_BACKENDS, "MoE"), (DENSE_BACKENDS, "Dense")):
        p(f"  {name}:")
        for be in grp:
            ok = be.exe.exists()
            p(f"    [{'✅' if ok else '❌'}] {be.key:<10} {be.label}")
    p()
    p(b("Tools:"))
    for t in (ROOT / "tools" / "reset-gpu.ps1", ROOT / "tools" / "adlx-profile.exe"):
        ok = t.exists()
        p(f"  [{'✅' if ok else '❌'}] {t}")


# ── CLI ───────────────────────────────────────────────────────────────────
def _add_model_opts(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--model", required=True,
                    help=" " .join(["one of: "]) + ", ".join(MODEL_BY_KEY))
    sp.add_argument("--backend", required=True)
    sp.add_argument("--ctx", type=int, required=True, choices=CTX_CHOICES)
    sp.add_argument("--vision", action="store_true", help="load mmproj (if model has one)")
    sp.add_argument("--mtp", action="store_true", help="enable MTP (if supported)")
    sp.add_argument("--mtp-mode", choices=["normal", "tuned"], default="tuned",
                    help="MTP preset: normal draft-mtp or model/backend-tuned (default)")
    sp.add_argument("--dflash", action="store_true",
                    help="enable DFlash2 draft (draft-dflash; Qwen3.8-27B models on b10701+ builds)")
    sp.add_argument("--port", type=int, default=DEFAULT_PORT)


def cmd_run(args: argparse.Namespace) -> None:
    m = find_model(args.model)
    b = next((x for x in backends_for(m) if x.key == args.backend), None)
    if b is None:
        raise SystemExit(f"Backend '{args.backend}' not valid for {m.key}: "
                         f"{[x.key for x in backends_for(m)]}")
    if args.vision and not m.mmproj:
        p(y(f"  {m.key} has no mmproj - vision ignored."))
    if args.mtp and not mtp_allowed(m, b):
        p(y(f"  MTP not supported on {b.key} for {m.key} - ignored."))
    dflash = args.dflash and dflash_allowed(m, b)
    if args.dflash and not dflash:
        p(y("  DFlash2 not supported here (Qwen3.8-27B + b10701+ build needed) - ignored."))
    rc = resolve(m, b, args.ctx, args.mtp and mtp_allowed(m, b),
                 args.vision and bool(m.mmproj), args.port, args.mtp_mode,
                 use_dflash=dflash)
    if args.dry_run:
        summarize(rc, dry=True)
        return
    run_server(rc)


def cmd_test(args: argparse.Namespace) -> None:
    m = find_model(args.model)
    b = next((x for x in backends_for(m) if x.key == args.backend), None)
    if b is None:
        raise SystemExit(f"Backend '{args.backend}' not valid for {m.key}: "
                         f"{[x.key for x in backends_for(m)]}")
    dflash = args.dflash and dflash_allowed(m, b)
    if args.dflash and not dflash:
        p(y("  DFlash2 not supported here (Qwen3.8-27B + b10701+ build needed) - ignored."))
    rc = resolve(m, b, args.ctx,
                 args.mtp and mtp_allowed(m, b),
                 args.vision and bool(m.mmproj), args.port, args.mtp_mode,
                 use_dflash=dflash)
    if args.dry_run:
        summarize(rc, dry=True)
        p(f"(test would POST /completion with n_tokens={args.tokens} prompt={args.prompt!r})")
        return
    run_test(rc, args.prompt, args.tokens, args.load_timeout)


def _cmd_gui() -> None:
    try:
        import run_model_ui
    except ImportError as e:
        raise SystemExit(f"Cannot import run_model_ui: {e}")
    run_model_ui.main()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Launcher & tester for llama-server (replaces run_model.bat). "
                    "With no subcommand, runs the interactive menu.")
    sub = parser.add_subparsers(dest="cmd")

    sp_run = sub.add_parser("run", help="start the server (like the bat)")
    _add_model_opts(sp_run)
    sp_run.add_argument("--dry-run", action="store_true", help="print the command only")
    sp_run.set_defaults(func=lambda a: cmd_run(a))

    sp_test = sub.add_parser("test", help="boot, measure pp/tg on a fixed prompt, stop")
    _add_model_opts(sp_test)
    sp_test.add_argument("--tokens", type=int, default=64)
    sp_test.add_argument("--prompt", default="Count from 1 to 100.")
    sp_test.add_argument("--load-timeout", type=float, default=300.0)
    sp_test.add_argument("--dry-run", action="store_true", help="print the command only")
    sp_test.set_defaults(func=lambda a: cmd_test(a))

    sp_gui = sub.add_parser("gui", help="open the graphical launcher (run_model_ui.py)")
    sp_gui.set_defaults(func=lambda a: _cmd_gui())

    sp_g = sub.add_parser("gpu-reset", help="PnP GPU reset (elevates via UAC if needed)")
    sp_g.set_defaults(func=lambda a: gpu_reset())

    sp_p = sub.add_parser("power", help="ADLX power profile")
    sp_p.add_argument("--profile", required=True, choices=["low", "stock", "max"])
    sp_p.set_defaults(func=lambda a: power_profile(a.profile))

    sp_l = sub.add_parser("list", help="list models/backends with on-disk status")
    sp_l.set_defaults(func=lambda a: list_all())

    _setup_console()
    args = parser.parse_args(argv)
    if args.cmd is None:
        main_menu()
        return
    args.func(args)


if __name__ == "__main__":
    main()
