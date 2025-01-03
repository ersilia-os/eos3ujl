import csv
from rdkit import Chem
from rdkit.Chem import Descriptors
from standardiser import standardise

with open("drugbank_smiles.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    smiles_list = []
    for r in reader:
        smiles_list.append(r[0])
    
std_smiles_list = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    try:
        mol = standardise.run(mol)
    except:
        print(f"Invalid SMILES: {smi}")
        continue
    mw = Descriptors.ExactMolWt(mol)
    if mw < 50:
        continue
    if mw > 1000:
        continue
    std_smiles_list.append(Chem.MolToSmiles(mol))

with open("drugbank_smiles_standardised.csv", "w") as f:
    f.write("SMILES\n")
    for smi in std_smiles_list:
        f.write(f"{smi}\n")