#!/usr/bin/env python3
"""Sum sizes of tensors whose name matches a prefix in a GGUF file (reads header only)."""
import struct, sys

def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def skip_value(f, vtype):
    # GGUF value types: 0=uint8,1=int8,2=uint16,3=int16,4=uint32,5=int32,6=float32,
    # 7=bool,8=string,9=array,10=uint64,11=int64,12=float64
    if vtype == 0: f.read(1)
    elif vtype == 1: f.read(1)
    elif vtype == 2: f.read(2)
    elif vtype == 3: f.read(2)
    elif vtype == 4: f.read(4)
    elif vtype == 5: f.read(4)
    elif vtype == 6: f.read(4)
    elif vtype == 7: f.read(1)
    elif vtype == 8: read_string(f)
    elif vtype == 9:
        atype = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n): skip_value(f, atype)
    elif vtype == 10: f.read(8)
    elif vtype == 11: f.read(8)
    elif vtype == 12: f.read(8)
    else: raise ValueError(f"unknown vtype {vtype}")

# GGUF quant type sizes in bytes per element (only the ones likely present)
SIZES = {0:1,1:1,2:1,3:2,4:4,5:4,6:4,7:2,8:2,9:4,10:4,11:4,12:4,13:8,14:8,
         15:8,16:8,17:2,18:2,19:2,20:4,21:4,22:4,23:8,24:2,25:2,26:2,27:4,28:4,
         29:4,30:4,31:8,32:8,33:8,34:8,35:8,36:8,37:8,38:8,39:8,40:8,41:4,42:4,
         43:4,44:4,45:4,46:4,47:8,48:8,49:4,50:4,51:8,52:8,53:8,54:8,55:8,56:8,
         57:8,58:8,59:8,60:8,61:8,62:4,63:8,64:8,65:4,66:4,67:4,68:4,69:8,70:8}
# block quant types: elements per block -> use n_blocks * block_size via dims

def block_size(t):
    # known block quant type sizes in bytes (ggml_type)
    B = {1:32,2:32,3:16,4:16,5:16,6:16,7:16,8:8,9:8,10:8,11:8,12:8,13:8,14:8,
         15:8,16:8,17:8,18:8,19:8,20:16,21:16,22:16,23:16,24:8,25:8,26:8,27:16,
         28:16,29:16,30:16,31:16,32:16,33:16,34:16,35:16,36:16,37:16,38:16,39:16,
         40:16,41:16,42:16,43:16,44:16,45:16,46:16,47:16,48:16,49:16,50:16,51:16,
         52:16,53:16,54:16,55:16,56:16,57:16,58:16,59:16,60:16,61:16,62:16,63:16,
         64:16,65:16,66:16,67:16,68:16,69:16,70:16}
    return B.get(t, 0)

def main(path, prefix):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"not gguf: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_kv):
            read_string(f)           # key
            skip_value(f, struct.unpack("<I", f.read(4))[0])
        # tensor infos
        total = 0
        match = 0
        n_match = 0
        for _ in range(n_tensors):
            name = read_string(f)
            n_dim = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack("<" + "Q"*n_dim, f.read(8*n_dim))
            ttype = struct.unpack("<I", f.read(4))[0]
            f.read(8)  # offset
            n_elems = 1
            for d in dims: n_elems *= d
            nbytes = n_elems * SIZES.get(ttype, 1)
            if ttype >= 1 and ttype <= 70:
                nbytes = (n_elems // 32) * block_size(ttype)
            total += nbytes
            if name.startswith(prefix):
                match += nbytes
                n_match += 1
        print(f"file={path}")
        print(f"tensors={n_tensors} total={total/1e9:.3f} GB")
        print(f"prefix '{prefix}': {n_match} tensors, {match/1e9:.3f} GB ({100*match/total:.2f}%)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
