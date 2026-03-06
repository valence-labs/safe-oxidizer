# `safe_oxidizer` — Python API Reference

`safe_oxidizer` is a Rust-backed Python extension. It provides two things: (1) SAFE encoding of SMILES strings, and (2) a BPE tokenizer trained on SAFE-encoded SMILES, with support for additional atomic tokens and BOS/EOS special tokens.

---

## SAFE Encoding

SAFE (Sequential Attachment-based Fragment Embedding) encodes a molecule as a dot-separated sequence of SMILES fragments, where BRICS bonds are cut and attachment points are represented as numbered ring-closure labels. The output is a valid SMILES string that decodes back to the original molecule via the Python `safe` library.

```python
import safe_oxidizer

# Single molecule → SAFE string. Raises ValueError on invalid SMILES.
safe_str = safe_oxidizer.safe_encode("Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1")

# Batch encoding (parallel via rayon). Returns List[Optional[str]]; None on failure.
results = safe_oxidizer.encode_batch(smiles_list)
results = safe_oxidizer.encode_batch(smiles_list, n_jobs=8)
```

**Notes:**
- Molecules with no BRICS bonds return the canonical SMILES of the molecule unchanged (single fragment, no dots).
- Aromatic output is kekulized (correct chemistry, may differ in aromaticity notation from input).
- E/Z stereo across BRICS-cut double bonds is lost (inherent to SAFE encoding, same as Python `safe` library).
- The Rust implementation is ~20x faster single-threaded than the Python `safe` library, with near-linear parallel scaling.

---

## `SafeTokenizer`

A BRICS-aware BPE tokenizer that first SAFE-encodes SMILES into fragments, then applies BPE merges within each fragment, joining fragments with a dot token.

### Token ID layout

```
0 – 255                         byte tokens
256 – 256+num_merges-1          BPE merge tokens
256+num_merges+0                DOT  (".")
256+num_merges+1                PAD  ("<pad>")
256+num_merges+2                BOS  ("<bos>")
256+num_merges+3                EOS  ("<eos>")
256+num_merges+4                UNK  ("<unk>")
256+num_merges+5 …              additional tokens (one ID per string, in insertion order)
```

### Construction and training

```python
from safe_oxidizer import SafeTokenizer

tok = SafeTokenizer()

# Train on an iterator of SMILES strings.
# vocab_size = 256 base bytes + BPE merges + 5 special tokens.
# Minimum vocab_size is 261 (zero merges).
tok.train_from_iterator(iter(smiles_list), vocab_size=4096)
tok.train_from_iterator(iter(smiles_list), vocab_size=4096, buffer_size=8192)  # default buffer
```

### Save / load

```python
tok.save("tokenizer.json")
tok = SafeTokenizer.load("tokenizer.json")  # static method
```

Old tokenizer files saved before `additional_tokens` existed load cleanly (field defaults to empty).

### Adding extra tokens (e.g. gene guide strings)

Each added string becomes a single atomic token — it bypasses BPE entirely and encodes as one ID. Useful for per-guide identifiers or any non-SMILES string that must be treated as a single token.

```python
tok.add_tokens(["ENSG00000141510", "ENSG00000157764", "guide_XYZ"])
# Deduplication: adding the same string twice is a no-op.
```

Added tokens are persisted by `save()` and restored by `load()`.

### Properties / getters

```python
tok.vocab_size             # int: total vocabulary size (includes additional tokens)
tok.num_additional_tokens  # int: number of tokens added via add_tokens()
tok.dot_token_id           # int
tok.pad_token_id           # int
tok.bos_token_id           # int
tok.eos_token_id           # int
tok.unk_token_id           # int
tok.get_vocab()            # List[Tuple[int, bytes]]: all (id, bytes) pairs
```

### Encoding

```python
# Single string → List[int]
ids = tok.encode("CCCOCC")
ids = tok.encode("CCCOCC", add_special_tokens=True)   # [bos, ..., eos]
ids = tok.encode("ENSG00000141510")                    # [<single additional token id>]

# Batch → numpy array of shape (N, L) dtype=uint32, padded with pad_token_id
arr = tok.batch_encode(smiles_list)
arr = tok.batch_encode(smiles_list, max_length=128)
arr = tok.batch_encode(smiles_list, shuffle=True)              # shuffles fragment order per sequence
arr = tok.batch_encode(smiles_list, add_special_tokens=True)   # wraps each row with BOS/EOS

# When add_special_tokens=True and max_length is set:
# inner tokens are truncated to (max_length - 2) first, then BOS/EOS are added,
# so the total row length is exactly max_length.
arr = tok.batch_encode(smiles_list, max_length=128, add_special_tokens=True)
```

**Behavior for additional tokens:** Any string registered via `add_tokens()` encodes to exactly one token ID regardless of content. The BPE/SAFE pipeline is never invoked for these strings.

**`shuffle=True`:** Randomizes the order of SAFE fragments before BPE, producing different token sequences for the same molecule on each call (same set of fragments, different ordering). Useful for data augmentation in generative models. Only affects SMILES inputs; additional tokens are unaffected.

### Decoding

```python
# Single sequence
text = tok.decode([256, 67, 45, ...])   # → string (SAFE-encoded SMILES or raw text)

# Batch
texts = tok.batch_decode([[...], [...], ...])  # → List[str]
```

Decoding concatenates the byte sequences for each token ID. Special token bytes (`<bos>`, `<eos>`, `<pad>`) will appear literally in the output if not stripped before calling decode.

---

## Typical usage pattern (generative model)

```python
tok = SafeTokenizer.load("tokenizer.json")

# Add per-perturbation guide tokens
tok.add_tokens(guide_ids)   # e.g. ["ENSG00000141510_guide1", ...]

# Encode a batch of (smiles, guide) pairs as separate sequences
smiles_tokens = tok.batch_encode(smiles_list, max_length=256, add_special_tokens=True)
guide_tokens  = tok.batch_encode(guide_ids,   max_length=1,   add_special_tokens=False)
```
