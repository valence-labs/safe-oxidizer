# safe-oxidizer

A pure Rust implementation of [SAFE](https://github.com/datamol-io/safe) (Sequential Attachment-based Fragment Embedding) encoding, exposed to Python via PyO3.

SAFE encodes molecules as dot-separated SMILES fragments with numbered attachment points, enabling sequential generation of molecules fragment-by-fragment. This crate reimplements `safe.encode()` in Rust for ~20x single-threaded speedup and near-linear parallel scaling.

## Installation

Install from the latest [GitHub release](https://github.com/valence-labs/safe-oxidizer/releases):

```bash
# With uv
uv pip install safe-oxidizer --find-links https://github.com/valence-labs/safe-oxidizer/releases/latest/download/

# With pip
pip install safe-oxidizer --find-links https://github.com/valence-labs/safe-oxidizer/releases/latest/download/
```

Or to pin a specific version:

```bash
uv pip install safe-oxidizer --find-links https://github.com/valence-labs/safe-oxidizer/releases/download/v0.1.0/
```

To add as a dependency in `pyproject.toml`:

```toml
[project]
dependencies = ["safe-oxidizer"]

[tool.uv]
find-links = ["https://github.com/valence-labs/safe-oxidizer/releases/latest/download/"]
```

## Usage

### SAFE encoding

```python
import safe_oxidizer

# Encode a single molecule
safe_str = safe_oxidizer.safe_encode("Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1")
# 'c15ccc(cc1)S(N)(=O)=O.Cc1ccc3cc1.c13n5nc4c1.C4(F)(F)F'

# Batch encode with parallelism
results = safe_oxidizer.encode_batch(smiles_list, n_jobs=8)  # Returns list[str | None]
```

### BPE tokenizer

The `SafeTokenizer` is a byte-pair encoding tokenizer that uses SAFE encoding as a pre-tokenization step — each SMILES is first split into BRICS fragments, then BPE merges are applied within each fragment.

```python
from safe_oxidizer import SafeTokenizer

# Train a tokenizer from an iterator of SMILES strings
tokenizer = SafeTokenizer()
tokenizer.train_from_iterator(iter(smiles_list), vocab_size=512)

# Encode a SMILES string to token IDs
ids = tokenizer.encode("CCO")

# Decode token IDs back to a SAFE string
safe_str = tokenizer.decode(ids)

# Batch encode (parallel) — returns a padded numpy array
token_array = tokenizer.batch_encode(smiles_list, max_length=128)

# Batch encode with fragment shuffling (for data augmentation)
token_array = tokenizer.batch_encode(smiles_list, max_length=128, shuffle=True)

# Batch decode
strings = tokenizer.batch_decode(token_array.tolist())

# Save / load
tokenizer.save("tokenizer.json")
tokenizer = SafeTokenizer.load("tokenizer.json")
```

Special tokens:

| Property | Description |
|---|---|
| `tokenizer.vocab_size` | Total vocabulary size |
| `tokenizer.dot_token_id` | Fragment separator (`.`) |
| `tokenizer.pad_token_id` | Padding token |
| `tokenizer.bos_token_id` | Beginning of sequence |
| `tokenizer.eos_token_id` | End of sequence |
| `tokenizer.unk_token_id` | Unknown token |

## Building from source

Requires a Rust toolchain.

```bash
uv tool install maturin
maturin develop --release --uv
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
├── lib.rs            # PyO3 module: safe_encode(), encode_batch(), SafeTokenizer
├── mol.rs            # Mol, Atom, Bond structs + sanitization
├── smiles_parser.rs  # SMILES string → Mol
├── smiles_writer.rs  # Mol → canonical SMILES string
├── brics.rs          # BRICS bond finding (predicate-based)
├── fragment.rs       # Bond cutting + connected components
├── encode.rs         # SAFE encoding algorithm
└── tokenizer.rs      # BPE tokenizer with SAFE pre-tokenization
```

## Testing

```bash
# Rust unit tests
cargo test

# Python tests (requires datamol, safe)
pytest tests/test_safe_oxidizer.py -v
```

## Known limitations

- E/Z stereo across BRICS-cut double bonds is lost (inherent to SAFE encoding)
- Aromatic SMILES output is kekulized (correct but differs from input form)
