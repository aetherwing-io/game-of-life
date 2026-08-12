#!/usr/bin/env python3
"""Deterministic verifier for life2 (16x16 animated, argv-parsing version).

Same architecture as verify.py: the generator's claims (claims2.json) are
validated by dumb arithmetic and byte comparison, then behavior is checked
against an independent Life implementation — including the argv paths.

  verify2.py check  <binary>
  verify2.py oracle <binary>
"""
import json
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = 16


def load_claims():
    return json.loads((HERE / "claims2.json").read_text())


def check(binpath):
    b = Path(binpath).read_bytes()
    claims = load_claims()
    lay = claims["layout"]
    code_off = claims["code_file_offset"]
    errors = []

    if b[:4] != b"\x7fELF":
        errors.append("bad ELF magic")
    entry = struct.unpack_from("<Q", b, 0x18)[0]
    if entry != lay["entry"]:
        errors.append(f"e_entry {entry:#x} != {lay['entry']:#x}")
    p_vaddr = struct.unpack_from("<Q", b, 0x40 + 16)[0]
    p_filesz = struct.unpack_from("<Q", b, 0x40 + 32)[0]
    p_memsz = struct.unpack_from("<Q", b, 0x40 + 40)[0]
    if p_vaddr != lay["base_vaddr"]:
        errors.append(f"p_vaddr {p_vaddr:#x}")
    if p_filesz != len(b):
        errors.append(f"p_filesz claims {p_filesz:#x} bytes but file is {len(b):#x}")
    if p_memsz != p_filesz:
        errors.append("p_memsz != p_filesz")
    if len(b) != lay["file_size"]:
        errors.append(f"file size {len(b)} != declared {lay['file_size']}")

    pos = 0
    code = bytearray()
    for ins in claims["instructions"]:
        raw = bytes.fromhex(ins["hex"])
        if ins["off"] != pos:
            errors.append(
                f"offset drift at {ins['hex']}: ledger {ins['off']}, cumulative {pos}"
            )
        jmp = ins.get("jmp")
        if jmp:
            size = jmp["size"]
            want = jmp["target"] - (pos + len(raw))
            got = int.from_bytes(raw[-size:], "little", signed=True)
            if got != want:
                errors.append(
                    f"jump at {pos}: encoded disp {got}, target {jmp['target']} needs {want}"
                )
            lo = -(1 << (8 * size - 1))
            if not lo <= want < -lo:
                errors.append(f"jump at {pos}: disp {want} overflows {size} byte(s)")
        code += raw
        pos += len(raw)

    if pos != claims["code_length"]:
        errors.append(f"ledger totals {pos}, claims code_length {claims['code_length']}")
    if b[code_off:code_off + len(code)] != bytes(code):
        actual = b[code_off:code_off + len(code)]
        i = next(i for i, (x, y) in enumerate(zip(code, actual)) if x != y)
        errors.append(f"code mismatch at code offset {i}: ledger {code[i]:02X}, file {actual[i]:02X}")
    if code_off + len(code) > lay["board_off"]:
        errors.append("code overruns board data")

    board = b[lay["board_off"]:lay["board_off"] + lay["board_size"]]
    if set(board) - {0, 1}:
        errors.append("board contains non-cell bytes")
    alive = {i for i, v in enumerate(board) if v == 1}
    if alive != set(lay["seed_cells"]):
        errors.append(f"board seed {sorted(alive)} != declared {sorted(lay['seed_cells'])}")
    if any(b[lay["next_off"]:lay["out_off"]]):
        errors.append("scratch region not zeroed")
    prefix = bytes.fromhex(lay["ansi_prefix_hex"])
    if b[lay["out_off"]:lay["out_off"] + len(prefix)] != prefix:
        errors.append("ANSI prefix missing from outbuf")
    if any(b[lay["out_off"] + len(prefix):lay["timespec_off"]]):
        errors.append("outbuf tail not zeroed")
    sec, nsec = struct.unpack_from("<QQ", b, lay["timespec_off"])
    if (sec, nsec) != (0, lay["timespec_nsec"]):
        errors.append(f"timespec ({sec},{nsec}) != (0,{lay['timespec_nsec']})")
    return errors


def reference_output(gens):
    lay = load_claims()["layout"]
    prefix = bytes.fromhex(lay["ansi_prefix_hex"]).decode()
    alive = {(c % N, c // N) for c in lay["seed_cells"]}
    out = []
    for _ in range(gens):
        out.append(prefix)
        for y in range(N):
            out.append("".join("#" if (x, y) in alive else "." for x in range(N)) + "\n")
        out.append("\n")
        nxt = set()
        for y in range(N):
            for x in range(N):
                n = sum(
                    ((x + dx) % N, (y + dy) % N) in alive
                    for dx in (-1, 0, 1)
                    for dy in (-1, 0, 1)
                    if (dx, dy) != (0, 0)
                )
                if n == 3 or (n == 2 and (x, y) in alive):
                    nxt.add((x, y))
        alive = nxt
    return "".join(out)


def oracle(binpath):
    lay = load_claims()["layout"]
    default = lay["default_generations"]
    cases = [([], default), (["1"], 1), (["3"], 3), (["10"], 10), (["abc"], default)]
    errors = []
    exe = str(Path(binpath).resolve())
    for args, gens in cases:
        label = f"argv={args or ['<none>']} -> {gens} gens"
        try:
            r = subprocess.run([exe] + args, capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            errors.append(f"{label}: timed out")
            continue
        if r.returncode != 0:
            errors.append(f"{label}: exit code {r.returncode}")
            continue
        ref = reference_output(gens)
        out = r.stdout.decode(errors="replace")
        if out != ref:
            frame_len = len(ref) // gens if gens else 0
            i = next((i for i, (a, e) in enumerate(zip(out, ref)) if a != e), None)
            if i is None:
                errors.append(f"{label}: length mismatch {len(out)} vs {len(ref)}")
            else:
                errors.append(f"{label}: first divergence at byte {i} (frame {i // frame_len})")
        else:
            print(f"  ok: {label}")
    return errors


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("check", "oracle"):
        print(__doc__)
        return 2
    mode, target = sys.argv[1], sys.argv[2]
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
