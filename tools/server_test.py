#!/usr/bin/env python3
"""Spawn llama-server, wait for health, run one completion, report pp/tg, kill.

Usage:
  python server_test.py --server D:/Work/llama/backends/bin/vulkan/llama-server.exe \
      -m "model.gguf" -c 32768 -b 4096 -ub 2048 -ncmoe 4 \
      --extra "--spec-type draft-mtp" -p 1000 -n 200 --health-timeout 60

Reports prompt_per_second / predicted_per_second from the server timings.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request

PROMPT_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the sphinx watches "
    "silently and the clock ticks onward into the long quiet night. "
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True, help="Path to llama-server.exe")
    p.add_argument("-m", "--model", required=True)
    p.add_argument("-c", "--ctx", type=int, default=32768)
    p.add_argument("-b", "--batch", type=int, default=4096)
    p.add_argument("-ub", "--ubatch", type=int, default=2048)
    p.add_argument("-ncmoe", "--ncmoe", type=int, default=0)
    p.add_argument("-p", "--prompt-tokens", type=int, default=1000)
    p.add_argument("-n", "--n-predict", type=int, default=200)
    p.add_argument("--extra", action="append", default=[],
                   help="Extra server flags, one per --extra (repeatable)")
    p.add_argument("--ctk", default="q4_0", help="KV cache type K (default: q4_0)")
    p.add_argument("--ctv", default="q4_0", help="KV cache type V (default: q4_0)")
    p.add_argument("--spec", default="", help="Speculative type, e.g. 'draft-mtp' (implies the flag)")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--health-timeout", type=int, default=60,
                   help="Seconds to poll /health before giving up")
    p.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature (default: 1.0)")
    return p.parse_args()


def build_prompt(target_tokens: int) -> str:
    # ~4.5 chars/token for English prose; overshoot then let server truncate.
    repeats = max(4, int(target_tokens * 4.5 / len(PROMPT_SENTENCE)))
    return PROMPT_SENTENCE * repeats


def wait_healthy(port: int, timeout: int) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    args = parse_args()
    cmd = [
        args.server, "-m", args.model, "-c", str(args.ctx),
        "-t", "8", "-b", str(args.batch), "--ubatch-size", str(args.ubatch),
        "-ctk", args.ctk, "-ctv", args.ctv, "--flash-attn", "on",
        "--load-mode", "none", "--n-cpu-moe", str(args.ncmoe),
        "--cache-ram", "0", "--no-repack", "--fit", "off",
        "-np", "1", "--host", "127.0.0.1", "--port", str(args.port),
        "-lv", "1",
    ]
    if args.spec:
        cmd += ["--spec-type", args.spec]
    for f in args.extra:
        cmd += f.split()
    print("CMD:", " ".join(cmd), file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)
    try:
        if not wait_healthy(args.port, args.health_timeout):
            stderr = proc.stderr.read() if proc.stderr else ""
            print("[FAIL] server never became healthy")
            print(stderr[-2000:])
            proc.kill()
            return 1
        prompt = build_prompt(args.prompt_tokens)
        payload = json.dumps({
            "prompt": prompt,
            "n_predict": args.n_predict,
            "cache_prompt": False,
            "temperature": args.temperature if args.temperature is not None else 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repeat_penalty": 1.0,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/completion", data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        t = data.get("timings", {})
        # Speculative decoding acceptance counters (from /slots).
        spec = {}
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{args.port}/slots", timeout=5) as r:
                slots = json.loads(r.read())
            if slots:
                s = slots[0]
                spec = {k: s.get(k) for k in
                        ("spec_accept", "spec_reject", "spec_n_drafted", "n_predict")}
        except Exception:
            pass
        print(f"[RESULT] prompt_n={t.get('prompt_n')} predicted_n={t.get('predicted_n')} "
              f"pp={t.get('prompt_per_second'):.1f} t/s tg={t.get('predicted_per_second'):.1f} t/s "
              f"tg_ms={t.get('predicted_ms'):.0f} ms spec={spec}", file=sys.stderr)
        print(json.dumps({
            "pp_tps": round(t.get("prompt_per_second", 0), 2),
            "tg_tps": round(t.get("predicted_per_second", 0), 2),
            "tg_ms_tok": round(t.get("predicted_ms", 0) / max(t.get("predicted_n", 1), 1), 2),
            "prompt_n": t.get("prompt_n"),
            "predicted_n": t.get("predicted_n"),
            "stop_reason": data.get("stop_reason"),
            "spec": spec,
        }))
        return 0
    finally:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, check=False)


if __name__ == "__main__":
    sys.exit(main())
