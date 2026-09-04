#!/usr/bin/env python3
"""Inject a terse system prompt into Qwen3.8's chat template + rewrite GGUF."""
import sys

import numpy as np
import gguf

SRC = r"C:/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q3_K_S.gguf"
DST = r"C:/Users/BhatSahab/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q3_K_S-terse.gguf"

TERSE = """Answer directly, after thinking. Lead with the answer, then only what it needs to be correct and usable.
Never: open with preamble or pleasantries; restate the question; add filler transitions; hedge with niceties; or repeat a point you've already made.
Always: keep essential steps, caveats, uncertainties, and specifics — never drop correctness or a needed warning for brevity. Keep the final answer lean. Use the least structure that conveys it (plain prose when short; lists or code only when they earn their place). If genuinely uncertain, say so and explain why — never omit uncertainty for the sake of brevity.
If a user request is genuinely ambiguous, ask a sharp question, don't guess."""

# User's snippet verbatim, plus the merged_system hook (this template's system var)
INJECTION = (
    "{%- set _terse %}\n"
    + TERSE
    + "\n{%- endset %}\n"
    "{%- if not _sc %}\n"
    "    {%- set _sc = _terse | trim %}\n"
    "{%- else %}\n"
    "    {%- set _sc = (_sc | trim) ~ '\\n\\n' ~ (_terse | trim) %}\n"
    "{%- endif %}\n"
    "{%- if reasoning_effort is not defined %}\n"
    "    {%- set reasoning_instructions = '' %}\n"
    "{%- endif %}\n"
    "{%- set merged_system = sysns.text %}\n"
    "{%- if merged_system %}\n"
    "    {%- set merged_system = (merged_system | trim) ~ '\\n\\n' ~ _sc %}\n"
    "{%- else %}\n"
    "    {%- set merged_system = _sc %}\n"
    "{%- endif %}"
)

ANCHOR = "{%- set merged_system = sysns.text %}"


def main():
    reader = gguf.GGUFReader(SRC)
    old_tpl = reader.fields["tokenizer.chat_template"].contents()
    assert ANCHOR in old_tpl, "anchor not found in template"
    new_tpl = old_tpl.replace(ANCHOR, INJECTION + "\n" + ANCHOR)
    assert new_tpl != old_tpl
    print(f"template: {len(old_tpl)} -> {len(new_tpl)} bytes")

    writer = gguf.GGUFWriter(DST, reader.fields["general.architecture"].contents())
    SKIP = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for name, field in reader.fields.items():
        if name in SKIP:
            continue
        vtype = field.types[0]
        val = field.contents()
        if name == "tokenizer.chat_template":
            val = new_tpl
        if vtype == gguf.GGUFValueType.ARRAY:
            writer.add_key_value(name, val, vtype, field.types[-1])
        else:
            writer.add_key_value(name, val, vtype)

    for t in reader.tensors:
        if t.data.dtype == np.uint8:
            writer.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
        else:
            writer.add_tensor(t.name, t.data, raw_shape=t.data.shape)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("done ->", DST)


if __name__ == "__main__":
    sys.exit(main())
