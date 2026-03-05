# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

safe-oxidizer is a high-performance Rust implementation of SAFE (Sequential Attachment-based Fragment Embedding) encoding, exposed to Python via PyO3. It encodes molecules as dot-separated SMILES fragments with numbered attachment points, enabling sequential fragment-by-fragment molecule generation. ~20x single-threaded speedup over Python with near-linear parallel scaling.

## Build & Development

```bash
source .venv/bin/activate          # Python env (managed with uv)
maturin develop --release          # Build and install Rust extension into venv
maturin develop                    # Debug build (faster compile, slower runtime)
```

## Testing

```bash
cargo test                                    # Rust unit tests
pytest tests/test_safe_oxidizer.py -v         # Python integration tests (needs datamol, safe)
pytest tests/test_safe_oxidizer.py -v -k test_name  # Single test
```

Tests validate chemical equivalence via `datamol.same_mol()` after roundtrip (Rust encode → Python decode), not string equality.

## Architecture

The encoding pipeline: SMILES string → `Mol` graph → find BRICS bonds → cut bonds → extract fragments → canonical SMILES per fragment → dot-joined output.

```
src/
├── lib.rs            # PyO3 module: safe_encode(), encode_batch()
├── encode.rs         # SAFE encoding orchestration
├── mol.rs            # Mol/Atom/Bond structs, sanitization, kekulization
├── smiles_parser.rs  # SMILES string → Mol
├── smiles_writer.rs  # Mol → canonical SMILES (Morgan-like invariant ranking)
├── brics.rs          # BRICS bond detection via 15 atom predicates × 46 pattern triples
└── fragment.rs       # Bond cutting + connected component extraction
```

### Key Design Decisions

- **BRICS without SMARTS**: The 46 BRICS patterns are decomposed into 15 atom-query predicates and matched as `(Q_left, Q_right, bond_type)` triples — no regex/SMARTS engine needed.
- **Full kekulization**: Augmenting-path matching for aromatic bond assignment with greedy fallback.
- **Chirality tracking**: Neighbor ordering + permutation counting (`count_swaps`) for @/@@.
- **Parallelism**: `rayon` thread pool via `encode_batch(smiles_list, n_jobs=N)`.

## Python API

```python
import safe_oxidizer

# Single molecule
result = safe_oxidizer.safe_encode("CCO")

# Batch (parallel)
results = safe_oxidizer.encode_batch(smiles_list, n_jobs=8)  # Returns Vec<Option<String>>
```

## Known Limitations

- E/Z stereo across BRICS-cut double bonds is lost (inherent to SAFE encoding)
- Aromatic SMILES output is kekulized (correct but differs from input form)
