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

## Honest caveats

This demonstrates the capability, not the practice. Hand-emitted machine code
is write-only, unauditable, and unforgiving — the first build here had a
2-byte padding miscount that silently shifted the data segment (and, by pure
luck, corrupted the glider into a still life). Toolchains exist for the humans
*and* for the machines: names, types, and relocations are what make code
checkable. But as a proof of concept that the translation layer is skippable —
here's the binary.
