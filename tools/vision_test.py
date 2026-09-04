#!/usr/bin/env python3
"""Spawn llama-server WITH vision (mmproj), wait for health, send an image
completion via /v1/chat/completions, report pp/tg, kill.

Usage:
  python vision_test.py --server D:/Work/llama/backends/bin/vulkan/llama-server.exe \
      -m "model.gguf" --mmproj "mmproj-F16.gguf" \
      -c 32768 -b 2048 -ub 2048 [--prompt "What color is this image?"] [--n 64]
"""

import argparse
import base64
import binascii
import json
import struct
import subprocess
import sys
import time
import urllib.request
import zlib


def make_png(path, w=64, h=64, rgb=(220, 40, 40)):
    """Minimal valid PNG (no dependency on PIL)."""
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", binascii.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    idat = zlib.compress(raw)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", idat))
        f.write(chunk(b"IEND", b""))


def wait_healthy(port, timeout):
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
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--mmproj", default="", help="mmproj path; omit for text-only chat completion")
    p.add_argument("-c", "--ctx", type=int, default=32768)
    p.add_argument("-b", "--batch", type=int, default=2048)
    p.add_argument("-ub", "--ubatch", type=int, default=2048)
    p.add_argument("-ncmoe", "--ncmoe", type=int, default=0)
    p.add_argument("--ctk", default="q4_0")
    p.add_argument("--ctv", default="q4_0")
    p.add_argument("--prompt", default="Describe this image. What color is the shape?")
    p.add_argument("--n", "--n-predict", type=int, default=64)
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--health-timeout", type=int, default=90)
    p.add_argument("--image", default="", help="Optional PNG path; else a generated red square")
    p.add_argument("--extra", action="append", default=[],
                   help="Extra server flags, one per --extra (repeatable). e.g. --extra=--spec-type --extra=draft-mtp")
    args = p.parse_args()

    img = args.image or "_vision_test.png"
    if not args.image:
        make_png(img)
    with open(img, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    uri = f"data:image/png;base64,{b64}"

    cmd = [
        args.server, "-m", args.model,
        "-c", str(args.ctx), "-t", "8",
        "-b", str(args.batch), "--ubatch-size", str(args.ubatch),
        "-ctk", args.ctk, "-ctv", args.ctv, "--flash-attn", "on",
        "--load-mode", "none", "--n-cpu-moe", str(args.ncmoe),
        "--cache-ram", "0", "--no-repack", "--fit", "off",
        "-np", "1", "--host", "127.0.0.1", "--port", str(args.port),
        "-lv", "1",
    ]
    if args.mmproj:
        cmd += ["--mmproj", args.mmproj]
    for f in args.extra:
        cmd += f.split()
    print("CMD:", " ".join(cmd), file=sys.stderr)
    logf = open("_vision_server.log", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, text=True)
    try:
        if not wait_healthy(args.port, args.health_timeout):
            logf.flush()
            with open("_vision_server.log", "r", encoding="utf-8", errors="replace") as f:
                print("[FAIL] server never became healthy")
                print(f.read()[-3000:])
            proc.kill()
            return 1

        payload = json.dumps({
            "model": "vision-test",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": uri}},
                ]}
            ],
            "max_tokens": args.n,
            "temperature": 1.0,
        }).encode()
        if not args.mmproj:
            payload = json.dumps({
                "model": "text-test",
                "messages": [{"role": "user", "content": args.prompt}],
                "max_tokens": args.n,
                "temperature": 1.0,
            }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            data=payload, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        elapsed = time.time() - t0

        content = ""
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            pass
        print("[DEBUG] message=", json.dumps(data["choices"][0]["message"])[:2000], file=sys.stderr)
        tim = data.get("timings", {})
        if tim:
            print(f"[RESULT] pp={tim.get('prompt_per_second'):.1f} "
                  f"tg={tim.get('predicted_per_second'):.1f} "
                  f"prompt_n={tim.get('prompt_n')} predicted_n={tim.get('predicted_n')}",
                  file=sys.stderr)
        print(json.dumps({
            "answer": content[:300],
            "elapsed_s": round(elapsed, 2),
            "pp_tps": round(tim.get("prompt_per_second", 0), 2),
            "tg_tps": round(tim.get("predicted_per_second", 0), 2),
            "prompt_n": tim.get("prompt_n"),
            "predicted_n": tim.get("predicted_n"),
            "stop_reason": data.get("stop_reason"),
        }))
        return 0
    except Exception as e:
        logf.flush()
        with open("_vision_server.log", "r", encoding="utf-8", errors="replace") as f:
            log = f.read()
        print(f"[FAIL] {e}")
        print("--- server log tail ---")
        print(log[-3000:])
        return 1
    finally:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, check=False)
        logf.close()


if __name__ == "__main__":
    sys.exit(main())
