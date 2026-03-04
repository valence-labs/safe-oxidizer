# safe-oxidizer

A pure Rust implementation of [SAFE](https://github.com/datamol-io/safe) (Sequential Attachment-based Fragment Embedding) encoding, exposed to Python via PyO3.

SAFE encodes molecules as dot-separated SMILES fragments with numbered attachment points, enabling sequential generation of molecules fragment-by-fragment. This crate reimplements `safe.encode()` in Rust for ~20x single-threaded speedup and near-linear parallel scaling.

## Quick start

```python
import safe_oxidizer

# Encode a single molecule
safe_str = safe_oxidizer.safe_encode("Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1")
# 'c15ccc(cc1)S(N)(=O)=O.Cc1ccc3cc1.c13n5nc4c1.C4(F)(F)F'

# Batch encode with parallelism
results = safe_oxidizer.encode_batch(smiles_list, n_jobs=8)
```

## Building

Requires a Rust toolchain.

```bash
# Install into a Python virtualenv
pip install maturin
maturin develop --release
```

## How it works

The encoder is a self-contained Rust implementation with no C/C++ dependencies:

1. **SMILES parser** — Parses input SMILES into an in-memory molecular graph (atoms, bonds, adjacency list). Handles the full organic subset, bracket atoms, chirality, stereochemistry, and ring closures.

2. **Sanitization** — Detects ring bonds via DFS, kekulizes aromatic systems using augmenting-path matching, and computes implicit hydrogens from valence rules.

3. **BRICS bond finding** — The 46 BRICS SMARTS patterns are decomposed into 15 atom-query predicates and 46 `(left, right, bond_type)` triples. Each bond is tested against these predicate pairs — no regex or SMARTS engine needed.

4. **Fragmentation** — Cuts identified bonds and inserts isotope-labeled dummy atoms (`[1*]`, `[2*]`, ...) at each cut site, then extracts connected components.

5. **SMILES writer** — Generates canonical SMILES for each fragment using Morgan-like invariant ranking. Fragments are sorted by size (largest first) and joined with dots.

6. **Attachment point numbering** — Replaces dummy atom labels with ring closure numbers, avoiding collisions with existing ring closures in the fragment SMILES.

## Architecture

```
src/
├── lib.rs            # PyO3 module: safe_encode(), encode_batch()
├── mol.rs            # Mol, Atom, Bond structs + sanitization
├── smiles_parser.rs  # SMILES string → Mol
├── smiles_writer.rs  # Mol → canonical SMILES string
├── brics.rs          # BRICS bond finding (predicate-based)
├── fragment.rs       # Bond cutting + connected components
└── encode.rs         # SAFE encoding algorithm
```

## Testing

```bash
# Rust unit tests
cargo test

# Python tests (requires datamol, safe)
pytest tests/test_safe_oxidizer.py -v
```

The Python test suite validates:
- Roundtrip correctness: encode with Rust, decode with Python `safe.decode()`, verify chemical equivalence via `datamol.same_mol()`
- Bracket atoms, charged species, isotopes, chirality, fused ring systems
- Batch encoding correctness and parallelism
- 10k-molecule chemical equivalence sweep
