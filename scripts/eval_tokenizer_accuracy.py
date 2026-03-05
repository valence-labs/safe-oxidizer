"""Evaluate fraction of SMILES correctly roundtripped through SafeTokenizer.

Pipeline: smiles -> tok.encode() -> tok.decode() -> safe_str
          then dm.same_mol(smiles, safe_str) for each pair.

Usage:
    python scripts/eval_tokenizer_accuracy.py
    python scripts/eval_tokenizer_accuracy.py --train-n 50000 --vocab-size 2000
"""
import argparse
import csv

import datamol as dm
from tqdm import tqdm

from safe_oxidizer import SafeTokenizer


def load_smiles(path: str, n: int | None) -> list[str]:
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        smiles = [row[0] for row in reader if row and row[0]]
    return smiles if n is None else smiles[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-csv", default="/mnt/ps/home/CORP/jason.hartford/project/safe/smiles.csv")
    parser.add_argument("--train-n", type=int, default=50_000, help="SMILES to train tokenizer on")
    parser.add_argument("--eval-n", type=int, default=None, help="SMILES to evaluate (default: all)")
    parser.add_argument("--vocab-size", type=int, default=1000)
    args = parser.parse_args()

    smiles_all = load_smiles(args.smiles_csv, args.eval_n)
    print(f"Loaded {len(smiles_all)} SMILES for evaluation")

    train_smiles = smiles_all[: args.train_n]
    print(f"Training tokenizer on {len(train_smiles)} SMILES with vocab_size={args.vocab_size}")
    tok = SafeTokenizer()
    tok.train_from_iterator(iter(train_smiles), vocab_size=args.vocab_size)
    print(f"Tokenizer trained: vocab_size={tok.vocab_size}")

    print(f"Encoding {len(smiles_all)} SMILES...")
    ids_list = [tok.encode(smi) for smi in tqdm(smiles_all)]
    safe_strings = tok.batch_decode(ids_list)

    correct = 0
    failures = []
    for smi, safe_str in tqdm(zip(smiles_all, safe_strings), total=len(smiles_all), desc="same_mol"):
        mol1 = dm.to_mol(smi)
        mol2 = dm.to_mol(safe_str)
        if dm.same_mol(mol1, mol2):
            correct += 1
        else:
            failures.append((smi, safe_str))

    frac = correct / len(smiles_all)
    print(f"\nResults: {correct}/{len(smiles_all)} correct ({frac:.2%})")

    if failures:
        print(f"\nFirst 10 failures:")
        for smi, safe_str in failures[:10]:
            print(f"  input:   {smi}")
            print(f"  decoded: {safe_str}")
            print()
    with open("failures.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "decoded"])
        for smi, safe_str in failures:
            writer.writerow([smi, safe_str])

if __name__ == "__main__":
    main()
