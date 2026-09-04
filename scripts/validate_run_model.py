#!/usr/bin/env python3
"""Static validation of run_model.py logic (no server is launched)."""
import run_model as rm

errors = []
checked = 0

def expect(cond, msg):
    global checked
    checked += 1
    if not cond:
        errors.append(msg)


# ── Config tables from run_model.bat ──────────────────────────────────────
CFG = {
    # model -> family -> ctx -> (batch, ubatch, ncmoe)
    "q36": {
        "vk":   {32768: (4096, 2048, 16), 65536: (4096, 2048, 16),
                 131072: (4096, 2048, 18), 262144: (512, 512, 21)},
        "rocm": {32768: (4096, 4096, 18), 65536: (4096, 4096, 18),
                 131072: (2048, 2048, 20), 262144: (512, 512, 20)},
    },
    "kat": {
        "vk":   {32768: (4096, 1024, 14), 65536: (4096, 1024, 14),
                 131072: (2048, 1024, 16), 262144: (512, 512, 18)},
        "rocm": {32768: (4096, 1024, 10), 65536: (4096, 1024, 11),
                 131072: (4096, 1024, 13), 262144: (512, 512, 17)},
    },
    "q38":  {"vk": {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                   131072: (512, 512, 0), 262144: (512, 512, 0)},
            "rocm": {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                     131072: (512, 512, 0), 262144: (512, 512, 0)}},
    "q38m": {"vk": {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                   131072: (512, 512, 0), 262144: (512, 512, 0)},
            "rocm": {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                     131072: (512, 512, 0), 262144: (512, 512, 0)}},
    "q8":   {"vk": {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                   131072: (512, 512, 0), 262144: (512, 512, 0)},
            "rocm": {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                     131072: (512, 512, 0), 262144: (512, 512, 0)}},
    "ridge": {"vk": {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                    131072: (1024, 2048, 0), 262144: (512, 512, 0)},
             "rocm": {32768: (2048, 2048, 0), 65536: (1024, 2048, 0),
                      131072: (1024, 2048, 0), 262144: (512, 512, 0)}},
}

Q38_SPEC = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1",
            "--spec-draft-p-min", "0.82",
            "--cache-type-k-draft", "q4_0", "--cache-type-v-draft", "q4_0"]
RIDGE_VK_SPEC = ["--spec-type", "draft-mtp,ngram-mod", "--spec-draft-p-min", "0.82",
                "--spec-draft-n-max", "2", "--cache-type-k-draft", "q4_0",
                "--cache-type-v-draft", "q4_0", "--spec-ngram-mod-n-match", "24",
                "--spec-ngram-mod-n-min", "8", "--spec-ngram-mod-n-max", "32"]
RIDGE_ROCM_SPEC = ["--spec-type", "draft-mtp-adaptive",
                  "--spec-draft-n-min-adaptive", "3", "--spec-draft-n-max", "8"]
PLAIN_SPEC = ["--spec-type", "draft-mtp"]
REASONING = ["--reasoning", "on", "--reasoning-preserve"]

MTP_ALLOWED = {  # model -> allowed backend keys
    "q36":  {"selfbuilt"},
    "kat":  {"vulkan", "selfbuilt", "stew", "vpatch", "rocm72"},
    "q38":  {"vulkan", "selfbuilt", "stew", "vpatch", "rocm72"},
    "q38m": {"vulkan", "selfbuilt", "stew", "vpatch", "rocm72"},
    "q8":   {"vulkan", "selfbuilt", "stew", "vpatch", "rocm72"},
    "ridge": {"vulkan", "selfbuilt", "stew", "vpatch", "rocm72"},
}
HAS_REASONING = {"q38", "q38m", "q8", "ridge"}
HAS_MMPROJ = {"q36", "q38", "q38m", "ridge"}
KV_DEFAULT = {"q8": ("q8_0", "q8_0")}

failures = []
for m in rm.MODELS:
    for be in rm.backends_for(m):
        for ctx in rm.CTX_CHOICES:
            for mtp in (False, True):
                for vision in (False, True):
                    rc = rm.resolve(m, be, ctx, mtp, vision, 8081)
                    key = f"{m.key}/{be.key}/{ctx}/mtp={mtp}/vis={vision}"
                    # batch table
                    if m.key == "ridge" and be.family == "vk" and mtp:
                        expect(rc.batch == 1024 and rc.ubatch == 128 and rc.ncpu_moe == 0,
                               f"{key}: ridge-vk-mtp special cfg, got {rc.batch}/{rc.ubatch}/{rc.ncpu_moe}")
                        expect(rc.kv == ["q4_1", "q4_1"], f"{key}: ridge-vk-mtp kv, got {rc.kv}")
                        expect(rc.cache_ram == 6000, f"{key}: ridge-vk-mtp cache-ram, got {rc.cache_ram}")
                        expect(rc.extra == ["--fit-target", "30", "--no-warmup",
                                            "--ctx-checkpoints", "96"],
                               f"{key}: ridge-vk-mtp extra, got {rc.extra}")
                    else:
                        exp = CFG[m.key][be.family][ctx]
                        expect((rc.batch, rc.ubatch, rc.ncpu_moe) == exp,
                               f"{key}: cfg, got ({rc.batch},{rc.ubatch},{rc.ncpu_moe}) want {exp}")
                        expect(rc.kv == list(KV_DEFAULT.get(m.key, ("q4_0", "q4_0"))),
                               f"{key}: kv, got {rc.kv}")
                        expect(rc.cache_ram == 0, f"{key}: cache-ram, got {rc.cache_ram}")
                        expect(rc.extra == [], f"{key}: extra, got {rc.extra}")
                    # MTP flags
                    if mtp and be.key in MTP_ALLOWED[m.key]:
                        want = {"q36": PLAIN_SPEC, "kat": PLAIN_SPEC, "q8": PLAIN_SPEC,
                                 "q38": Q38_SPEC, "q38m": Q38_SPEC}[m.key] if m.key in ("q36", "kat", "q8", "q38", "q38m") \
                                 else RIDGE_VK_SPEC if be.family == "vk" else RIDGE_ROCM_SPEC
                        expect(rc.spec == want, f"{key}: spec flags, got {rc.spec}")
                    else:
                        expect(rc.spec == [], f"{key}: spec should be off, got {rc.spec}")
                    # vision
                    if vision and m.key in HAS_MMPROJ and be.key != "stew":
                        expect(rc.mmproj == ["--mmproj", str(m.mmproj)],
                               f"{key}: mmproj, got {rc.mmproj}")
                    else:
                        expect(rc.mmproj == [], f"{key}: mmproj should be off, got {rc.mmproj}")
                    # command assembly
                    cmd = rm.build_cmd(rc)
                    expect(cmd[0].endswith("llama-server.exe"), f"{key}: exe path")
                    expect("-m" in cmd and cmd[cmd.index("-m") + 1] == str(m.path), f"{key}: -m")
                    expect("-c" in cmd and cmd[cmd.index("-c") + 1] == str(ctx), f"{key}: -c")
                    for flag, val in (("-b", rc.batch), ("--ubatch-size", rc.ubatch),
                                       ("--n-cpu-moe", rc.ncpu_moe),
                                       ("--cache-ram", rc.cache_ram)):
                        expect(flag in cmd and cmd[cmd.index(flag) + 1] == str(val),
                               f"{key}: {flag}={val} missing")
                    expect("--jinja" in cmd, f"{key}: --jinja")
                    if m.key in HAS_REASONING:
                        idx = cmd.index("--reasoning")
                        expect(cmd[idx:idx + 3] == REASONING, f"{key}: reasoning flags, tail={cmd[-4:]}")
                        expect("LLAMA_ARG_CHAT_TEMPLATE_KWARGS" in rm.launch_env(rc),
                               f"{key}: chat kwargs env")
                    else:
                        expect("--reasoning" not in cmd, f"{key}: unexpected reasoning")
                        expect("LLAMA_ARG_CHAT_TEMPLATE_KWARGS" not in rm.launch_env(rc),
                               f"{key}: unexpected chat kwargs env")
                    # sampling flags always present
                    for s in rm.SAMPLING:
                        expect(s in cmd, f"{key}: sampling flag {s} missing")
                    checked += 0  # counted above

# Normal MTP uses plain draft-mtp for every allowed model/backend pair.
for m in rm.MODELS:
    for be in rm.backends_for(m):
        rc = rm.resolve(m, be, 32768, True, False, 8081, "normal")
        key = f"{m.key}/{be.key}/normal"
        if be.key in MTP_ALLOWED[m.key]:
            expect(rc.spec == rm.normal_mtp_flags(m), f"{key}: normal MTP flags, got {rc.spec}")
            if m.key == "ridge" and be.family == "vk":
                exp = CFG[m.key][be.family][32768]
                expect((rc.batch, rc.ubatch, rc.ncpu_moe) == exp,
                       f"{key}: normal MTP must retain base config")
        else:
            expect(rc.spec == [], f"{key}: unsupported normal MTP must be off")

# MTP gating function
for mk, allowed in MTP_ALLOWED.items():
    for be in rm.backends_for(rm.MODEL_BY_KEY[mk]):
        expect(rm.mtp_allowed(rm.MODEL_BY_KEY[mk], be) == (be.key in allowed),
               f"mtp_allowed({mk}, {be.key})")

# backend groups
expect([b.key for b in rm.DENSE_BACKENDS] == ["vulkan", "stew", "selfbuilt"], "dense backends")
expect([b.key for b in rm.MOE_BACKENDS] == ["vulkan", "selfbuilt", "stew", "vpatch", "rocm72"], "moe backends")
for m in rm.MODELS:
    expect(rm.backends_for(m) is rm.DENSE_BACKENDS if m.is_dense else rm.MOE_BACKENDS,
           f"backends_for({m.key})")

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
print(f"Checked {checked} assertions.")
if errors:
    print(f"FAILED: {len(errors)} error(s):")
    for e in errors[:60]:
        print("  x", e)
else:
    print("ALL CHECKS PASSED")
