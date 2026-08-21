#!/usr/bin/env python3
"""Deterministic verifier for the hand-written `life` binary.

The division of labor under test: a probabilistic generator (the LLM) emits
the binary plus claims about it (claims.json); this small, dumb, deterministic
checker validates the claims. It knows nothing about x86 semantics — it only
does arithmetic and byte comparison, which is exactly why its failures are
uncorrelated with the generator's.

  verify.py check  <binary>   structural check against claims.json + ELF invariants
  verify.py oracle <binary>   differential test vs an independent Life implementation
  verify.py mutate <binary>   flip each code byte once; classify how corruption fails
"""
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_VADDR = 0x400000
BOARD_OFF, NEXT_OFF, OUT_OFF = 0x200, 0x240, 0x280
FILE_SIZE = 0x2C9
GLIDER_CELLS = {10, 19, 25, 26, 27}
GENERATIONS = 16


# ---------------------------------------------------------------- structural
def check(binpath):
    b = Path(binpath).read_bytes()
    claims = json.loads((HERE / "claims.json").read_text())
    code_off = claims["code_file_offset"]
    errors = []

    # ELF header invariants
    if b[:4] != b"\x7fELF":
        errors.append("bad ELF magic")
    entry = struct.unpack_from("<Q", b, 0x18)[0]
    if entry != BASE_VADDR + code_off:
        errors.append(f"e_entry {entry:#x} != {BASE_VADDR + code_off:#x}")
    p_vaddr = struct.unpack_from("<Q", b, 0x40 + 16)[0]
    p_filesz = struct.unpack_from("<Q", b, 0x40 + 32)[0]
    p_memsz = struct.unpack_from("<Q", b, 0x40 + 40)[0]
    if p_vaddr != BASE_VADDR:
        errors.append(f"p_vaddr {p_vaddr:#x} != {BASE_VADDR:#x}")
    if p_filesz != len(b):
        errors.append(f"p_filesz claims {p_filesz:#x} bytes but file is {len(b):#x}")
    if p_memsz != p_filesz:
        errors.append(f"p_memsz {p_memsz:#x} != p_filesz {p_filesz:#x}")
    if len(b) != FILE_SIZE:
        errors.append(f"file size {len(b)} != declared layout size {FILE_SIZE}")

    # Instruction ledger: offsets must be cumulative, bytes must match the
    # binary, and every declared jump displacement must equal target - next.
    pos = 0
    code = bytearray()
    for ins in claims["instructions"]:
        raw = bytes.fromhex(ins["hex"])
        if ins["off"] != pos:
            errors.append(
                f"offset drift at {ins['hex']}: ledger says {ins['off']}, "
                f"cumulative length says {pos}"
            )
        jmp = ins.get("jmp")
        if jmp:
            size = jmp["size"]
            want = jmp["target"] - (pos + len(raw))
            got = int.from_bytes(raw[-size:], "little", signed=True)
            if got != want:
                errors.append(
                    f"jump at {pos}: encoded disp {got}, but target "
                    f"{jmp['target']} needs {want}"
                )
            lo = -(1 << (8 * size - 1))
            if not lo <= want < -lo:
                errors.append(f"jump at {pos}: disp {want} overflows {size} byte(s)")
        code += raw
        pos += len(raw)

    if pos != claims["code_length"]:
        errors.append(f"ledger totals {pos} bytes, claims code_length {claims['code_length']}")
    actual = b[code_off:code_off + len(code)]
    if actual != bytes(code):
        i = next(i for i, (x, y) in enumerate(zip(code, actual)) if x != y)
        errors.append(f"code mismatch at code offset {i}: ledger {code[i]:02X}, file {actual[i]:02X}")
    if code_off + len(code) > BOARD_OFF:
        errors.append("code overruns board data")

    # Data landmarks
    board = b[BOARD_OFF:BOARD_OFF + 64]
    if len(board) == 64:
        if set(board) - {0, 1}:
            errors.append("board contains non-cell bytes")
        alive = {i for i, v in enumerate(board) if v == 1}
        if alive != GLIDER_CELLS:
            errors.append(f"board seed {sorted(alive)} != glider {sorted(GLIDER_CELLS)}")
    else:
        errors.append("board data truncated")
    if any(b[NEXT_OFF:FILE_SIZE]):
        errors.append("scratch/output regions not zeroed")

    return errors


# -------------------------------------------------- independent semantic oracle
def reference_output(gens=GENERATIONS):
    # Deliberately different representation from the binary: a set of (x, y)
    # tuples rather than a flat byte array.
    alive = {(2, 1), (3, 2), (1, 3), (2, 3), (3, 3)}
    lines = []
    for _ in range(gens):
        for y in range(8):
            lines.append("".join("#" if (x, y) in alive else "." for x in range(8)))
        lines.append("")
        nxt = set()
        for y in range(8):
            for x in range(8):
                n = sum(
                    ((x + dx) % 8, (y + dy) % 8) in alive
                    for dx in (-1, 0, 1)
                    for dy in (-1, 0, 1)
                    if (dx, dy) != (0, 0)
                )
                if n == 3 or (n == 2 and (x, y) in alive):
                    nxt.add((x, y))
        alive = nxt
    return "\n".join(lines) + "\n"


def oracle(binpath):
    ref = reference_output()
    r = subprocess.run([str(Path(binpath).resolve())], capture_output=True, timeout=5)
    errors = []
    if r.returncode != 0:
        errors.append(f"exit code {r.returncode}")
    out = r.stdout.decode(errors="replace")
    if out != ref:
        for i, (a, e) in enumerate(zip(out.splitlines(), ref.splitlines())):
            if a != e:
                errors.append(f"first divergence at line {i}: got {a!r}, expected {e!r}")
                break
        else:
            errors.append(f"length mismatch: got {len(out)} bytes, expected {len(ref)}")
    return errors


# ------------------------------------------------------------- mutation sweep
def mutate(binpath):
    claims = json.loads((HERE / "claims.json").read_text())
    code_off = claims["code_file_offset"]
    code_len = claims["code_length"]
    orig = Path(binpath).read_bytes()
    ref = reference_output().encode()
    counts = {"crash": 0, "hang": 0, "silent_wrong": 0, "neutral": 0}
    silent_examples = []

    scratch = os.environ.get("TMPDIR", tempfile.gettempdir())
    for pos in range(code_off, code_off + code_len):
        m = bytearray(orig)
        m[pos] ^= 0xFF  # deterministic single-byte corruption
        with tempfile.NamedTemporaryFile(dir=scratch, delete=False) as f:
            f.write(m)
            path = f.name
        os.chmod(path, stat.S_IRWXU)
        try:
            r = subprocess.run([path], capture_output=True, timeout=1)
            if r.returncode != 0:
                counts["crash"] += 1
            elif r.stdout == ref:
                counts["neutral"] += 1
            else:
                counts["silent_wrong"] += 1
                if len(silent_examples) < 5:
                    silent_examples.append(pos - code_off)
        except subprocess.TimeoutExpired:
            counts["hang"] += 1
        finally:
            os.unlink(path)
    return counts, silent_examples


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("check", "oracle", "mutate"):
        print(__doc__)
        return 2
    mode, target = sys.argv[1], sys.argv[2]
    if mode == "mutate":
        counts, examples = mutate(target)
        total = sum(counts.values())
        print(f"mutated {total} code bytes (one XOR-0xFF flip each):")
        for k, v in counts.items():
            print(f"  {k:13s} {v:4d}  ({100 * v / total:.0f}%)")
        print(f"exit code alone misses {counts['silent_wrong'] + counts['neutral']} of {total} "
              f"({100 * (counts['silent_wrong'] + counts['neutral']) / total:.0f}%); "
              f"the oracle catches every non-neutral one")
        if examples:
            print(f"example silently-wrong mutation offsets: {examples}")
        return 0
    errors = check(target) if mode == "check" else oracle(target)
    if errors:
        print(f"FAIL {mode} {target}")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"PASS {mode} {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
