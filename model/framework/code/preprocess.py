import sys
import csv
import os

root = os.path.dirname(os.path.abspath(__file__))

input_file = sys.argv[1]

smiles_list = []
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for r in reader:
        smiles_list.append(r[0])

output_file = os.path.join(root, "..", "Sample_Input.csv")
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["SMILES"])  # header
    for s in smiles_list:
        writer.writerow([s])