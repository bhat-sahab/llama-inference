#!/usr/bin/env python3
"""One-shot: spawn server w/ log capture, wait health, one completion, print log tail."""
import argparse, json, subprocess, sys, time, urllib.request

p = argparse.ArgumentParser()
p.add_argument("--server", required=True)
p.add_argument("-m", required=True)
p.add_argument("--port", type=int, default=8081)
p.add_argument("-n", type=int, default=200)
a = p.parse_args()

cmd = [a.server, "-m", a.m, "-c", "32768", "-t", "8", "-b", "2048",
       "--ubatch-size", "2048", "-ctk", "q4_0", "-ctv", "q4_0",
       "--flash-attn", "on", "--load-mode", "none", "--n-cpu-moe", "0",
       "--cache-ram", "0", "--no-repack", "--fit", "off", "-np", "1",
       "--host", "127.0.0.1", "--port", str(a.port), "-lv", "1"]
log = open("smoke.log", "w", encoding="utf-8", errors="replace")
print("CMD:", " ".join(cmd), file=sys.stderr)
proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
try:
    ok = False
    for _ in range(240):
        if proc.poll() is not None:
            print("[FAIL] server exited early, code", proc.returncode, file=sys.stderr)
            break
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)
            ok = True
            break
        except Exception:
            time.sleep(1)
    if not ok:
        print("[FAIL] never healthy", file=sys.stderr)
        sys.exit(1)
    print("[OK] healthy", file=sys.stderr)
    sent = "The quick brown fox jumps over the lazy dog while the sphinx watches silently. " * 110
    payload = json.dumps({"prompt": sent, "n_predict": a.n, "cache_prompt": False,
                          "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                          "min_p": 0.0, "presence_penalty": 0.0, "repeat_penalty": 1.0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/completion", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    t = data.get("timings", {})
    print(f"[RESULT] pp={t.get('prompt_per_second'):.1f} tg={t.get('predicted_per_second'):.1f} "
          f"prompt_n={t.get('prompt_n')} predicted_n={t.get('predicted_n')} "
          f"stop={data.get('stop_reason')}", file=sys.stderr)
    print(json.dumps({"pp": t.get("prompt_per_second"), "tg": t.get("predicted_per_second"),
                      "prompt_n": t.get("prompt_n"), "predicted_n": t.get("predicted_n"),
                      "stop": data.get("stop_reason")}))
finally:
    proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    log.close()
    print("--- LOG TAIL ---", file=sys.stderr)
    with open("smoke.log", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print("".join(lines[-30:]), file=sys.stderr)
