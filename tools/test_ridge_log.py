#!/usr/bin/env python3
"""Spawn server w/ log reader thread -> health -> dump load lines -> run completion."""
import json
import subprocess
import sys
import threading
import time
import urllib.request

SERVER = "D:/Work/llama/backends/bin/vulkan-new/llama-server.exe"
MODEL = "C:/Users/BhatSahab/.lmstudio/models/empero-ai/Qwen3.8-27B-Ridge-GGUF/Qwen3.8-27B-Ridge-3.7bpw.gguf"
PORT = 8085
CTX = sys.argv[1] if len(sys.argv) > 1 else "131072"

cmd = [
    SERVER, "-m", MODEL, "-c", CTX, "-t", "8", "-b", "1024", "--ubatch-size", "2048",
    "-ctk", "q4_0", "-ctv", "q4_0", "--flash-attn", "on", "--load-mode", "none",
    "--n-cpu-moe", "0", "--cache-ram", "0", "--no-repack", "--fit", "off",
    "-np", "1", "--host", "127.0.0.1", "--port", str(PORT), "-lv", "3",
    "--reasoning", "on", "--reasoning-preserve", "--jinja", "-ngl", "99",
]
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
lines = []
def reader():
    for ln in proc.stderr:
        lines.append(ln.rstrip())
threading.Thread(target=reader, daemon=True).start()

try:
    deadline = time.time() + 90
    ok = False
    while time.time() < deadline:
        if proc.poll() is not None:
            print("[FAIL] exited rc=%s" % proc.returncode)
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    ok = True
                    break
        except Exception:
            pass
        time.sleep(1)
    print("healthy:", ok)
    for pat in ("offload", "VRAM", "buffer", "kv", "backend", "tensor", "warn", "error", "failed", "mem", "layer"):
        for ln in lines:
            if pat in ln.lower():
                print("LOG|", ln)
    if not ok:
        sys.exit(1)
    s = ("The quick brown fox jumps over the lazy dog while the sphinx watches "
         "silently and the clock ticks onward into the long quiet night. ") * 220
    payload = json.dumps({
        "prompt": s, "n_predict": 100, "cache_prompt": False,
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 0.0, "repeat_penalty": 1.0,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=payload,
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    t = d.get("timings", {})
    print("pp", round(t.get("prompt_per_second", 0), 1),
          "tg", round(t.get("predicted_per_second", 0), 1),
          "pred_n", t.get("predicted_n"))
finally:
    proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    time.sleep(1)
    for ln in lines:
        if "offload" in ln.lower() or "buffer size" in ln.lower():
            print("LOG-LATE|", ln)
