#!/usr/bin/env python3
"""Read selected GGUF scalar metadata (no deps)."""
import struct
import sys

MAGIC = b"GGUF"


def read_gguf_kv(path, wanted):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC, f"not GGUF: {magic}"
        (version,) = struct.unpack("<I", f.read(4))
        (tensor_count,) = struct.unpack("<Q", f.read(8))
        (kv_count,) = struct.unpack("<Q", f.read(8))

        def read_string():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", errors="replace")

        def read_value(vtype):
            if vtype == 0:
                return struct.unpack("<B", f.read(1))[0]
            if vtype == 1:
                return struct.unpack("<b", f.read(1))[0]
            if vtype == 2:
                return struct.unpack("<H", f.read(2))[0]
            if vtype == 3:
                return struct.unpack("<h", f.read(2))[0]
            if vtype == 4:
                return struct.unpack("<I", f.read(4))[0]
            if vtype == 5:
                return struct.unpack("<i", f.read(4))[0]
            if vtype == 6:
                return struct.unpack("<f", f.read(4))[0]
            if vtype == 7:
                return bool(struct.unpack("<B", f.read(1))[0])
            if vtype == 8:
                return read_string()
            if vtype == 10:
                return struct.unpack("<Q", f.read(8))[0]
            if vtype == 11:
                return struct.unpack("<q", f.read(8))[0]
            if vtype == 12:
                return struct.unpack("<d", f.read(8))[0]
            if vtype == 9:  # array
                (atype,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                return [read_value(atype) for _ in range(n)]
            raise ValueError(f"unknown type {vtype}")

        found = {}
        for _ in range(kv_count):
            key = read_string()
            (vtype,) = struct.unpack("<I", f.read(4))
            val = read_value(vtype)
            if key in wanted:
                found[key] = val
        return found


if __name__ == "__main__":
    wanted = sys.argv[2].split(",") if len(sys.argv) > 2 else [
        "general.architecture", "qwen35moe.block_count", "qwen35moe.expert_count",
        "qwen35moe.expert_used_count", "qwen35moe.expert_feed_forward_length",
        "qwen35moe.expert_shared_feed_forward_length",
    ]
    result = read_gguf_kv(sys.argv[1], wanted)
    for k, v in result.items():
        print(f"{k} = {v}")
