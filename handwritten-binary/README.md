# Hand-Written Binary: Game of Life with No Compiler

An experiment: can an LLM skip the human-to-machine translation layer
(languages, compilers, assemblers, linkers) and write a working binary
executable directly? Answer: yes, crudely.

`life` is a 713-byte statically-linked x86-64 Linux ELF executable that runs
Conway's Game of Life — a glider on an 8×8 torus — and prints 16 generations
to stdout. Every byte of it was authored by hand as hex in `life.hex`:

- the ELF header and program header, field by field
- ~80 machine instructions, with REX prefixes, ModRM/SIB bytes, and all
  relative jump displacements encoded and offset-computed by hand
- the seed data (a glider) at hand-chosen virtual addresses

No toolchain touches it. The only "build step" is a character-set conversion
from hex digits to the bytes they denote:

```sh
sed 's/#.*//' life.hex | tr -d ' \n' | \
  python3 -c "import sys; sys.stdout.buffer.write(bytes.fromhex(sys.stdin.read()))" > life
chmod +x life
./life
```

(`sed 's/#.*//' life.hex | xxd -r -p > life` works too, where `xxd` exists.)

## How it works

One RWX `PT_LOAD` segment maps the whole file at `0x400000`. The entry point
(`0x400078`, right after the two headers) runs a loop of: render the 64-cell
board into an output buffer (`.`/`#` plus newlines), `write(2)` it to stdout,
compute the next generation with the classic B3/S23 rules (neighbour
coordinates wrap via `and 7`, so the board is a torus), copy it back, repeat
16 times, then `exit(0)`. It talks to the kernel only through raw `syscall`
instructions — no libc, no sections, no symbols, no relocations.

## The verification experiment

If models are ever to emit binaries directly, the interesting question is not
fluency but error detection — so this directory also tests the "probabilistic
generator + small deterministic checker" architecture on itself:

- **`claims.json`** — the instruction ledger the generator produced while
  hand-assembling: every instruction's offset and bytes, and every jump's
  intended target. Proof-carrying code in miniature.
- **`verify.py check life`** — a deliberately dumb checker (byte comparison and
  arithmetic only, no x86 knowledge) that validates the ledger's internal
  consistency, the jump displacements, the ELF header invariants, and the data
  landmarks.
- **`verify.py oracle life`** — differential test against an independent Life
  implementation (different data representation, same rules).
- **`verify.py mutate life`** — flips each of the 249 code bytes in turn and
  classifies how the corrupted binary fails.

Results on this binary:

1. Both checks PASS on the good binary.
2. Reconstructing the original 2-byte padding bug: the buggy binary **exits 0**
   — but `check` flags the exact cause (`p_filesz` 0x2C9 vs actual 0x2C7, board
   seed shifted) and `oracle` flags the symptom (wrong output from line 1).
3. Mutation sweep over all 249 code bytes: **72% crash, 8% hang, 19% run to
   completion with exit 0 and wrong output, 1% are neutral.** Exit code alone
   misses one in five corruptions; the output oracle catches every non-neutral
   one.

That 19% is the empirical version of the claim above: machine code has no
redundancy to make errors loud, so a generator at this level *needs* an
independent verifier — and a ~200-line deterministic one is enough to catch
both the real bug from this binary's development and every silent mutant.

## Round two: life2, and the token economics

`life2.hex` raises the difficulty: 16×16 torus, an R-pentomino (chaotic
evolution), ANSI clear-screen animation, a 60 ms `nanosleep` between frames,
and argv parsing (`./life2 N` runs N generations) — meaning stack access at
the entry point and a hand-rolled `atoi`, ~100 instructions and 21 hand-computed
jump displacements. This time the ledger (`claims2.json`) and checker
(`verify2.py`) were written *before* first execution, and the binary passed
the structural check and all five oracle cases (default, numeric, and invalid
argv) on the first build.

`life2.c` is the behavior-identical C program (verified against the same
oracle) for comparing costs:

| artifact                          | chars  | ~output tokens |
|-----------------------------------|--------|----------------|
| `life2.c` (traditional pipeline)  | 1,511  | ~380           |
| raw binary payload (1,320 bytes)  | —      | ~1,320         |
| `life2.hex` as actually emitted   | 12,391 | ~2,610         |
| `claims2.json` ledger             | 4,768  | ~1,190         |

(Token counts are estimates: ~1 token per space-separated hex byte, ~4 chars
per token for source. )

So direct emission cost ~3.5× the C version in pure payload tokens, ~7× as
emitted with annotations, ~10× with the verification ledger — before counting
the much larger *reasoning* overhead of offset bookkeeping, which the C
version simply doesn't have. The edit asymmetry is worse than the emission
asymmetry: inserting one instruction mid-program shifts every later offset and
invalidates every displacement across the gap — an O(program) re-emission —
where a C edit is local.

The one economic win for direct emission: the deliverable. The hand-written
binary is 1,320 bytes; `gcc -O2` produces 16 KB dynamic / 785 KB static for
identical behavior.

## Honest caveats

This demonstrates the capability, not the practice. Hand-emitted machine code
is write-only, unauditable, and unforgiving — the first build here had a
2-byte padding miscount that silently shifted the data segment (and, by pure
luck, corrupted the glider into a still life). Toolchains exist for the humans
*and* for the machines: names, types, and relocations are what make code
checkable. But as a proof of concept that the translation layer is skippable —
here's the binary.
