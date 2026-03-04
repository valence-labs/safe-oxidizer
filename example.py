import safe_oxidizer
import time
import csv

SMILES_CSV='/mnt/ps/home/CORP/jason.hartford/project/safe/smiles.csv' # list of 100k smiles
with open(SMILES_CSV) as f:
    reader = csv.reader(f)
    next(reader) # skip header
    print("loading 100k smiles from disk - this might take a while")
    smiles_list = [row[0] for row in reader if row and row[0]]

print("Starting batch processing")
t = time.time()
safe_batch = safe_oxidizer.encode_batch(smiles_list, n_jobs=24)
print(f"Done in {time.time() - t} secs")
