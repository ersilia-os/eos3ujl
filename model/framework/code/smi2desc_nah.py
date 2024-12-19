import sys
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd

def calculate_descriptors_from_csv(input_csv):
    # Read CSV file containing SMILES strings
    df = pd.read_csv(input_csv)
    
    # Check if "SMILES" column exists
    if "SMILES" not in df.columns:
        raise ValueError("CSV file must contain a 'SMILES' column")

    # Convert SMILES to RDKit molecule objects
    df['Molecule'] = df['SMILES'].apply(Chem.MolFromSmiles)

    # Initialize Mordred calculator with all descriptors, ignoring 3D descriptors
    calc = Calculator(descriptors, ignore_3D=False)

    # Calculate descriptors
    result = calc.pandas(df['Molecule'])

    # Merge original SMILES with their descriptors for clarity
    result.insert(0, 'SMILES', df['SMILES'])

    # Rename the 'nAHRing' column to 'nAHRing.1' if it exists in the DataFrame
    if 'nAHRing' in result.columns:
        result.rename(columns={'nAHRing': 'nAHRing.1'}, inplace=True)

    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv_file>")
        sys.exit(1)
    
    input_csv = sys.argv[1]  # Get the CSV file path from command line arguments
    try:
        descriptors_df = calculate_descriptors_from_csv(input_csv)
        print(descriptors_df.head())  # Print the first few rows of the DataFrame
        # Optionally, save the result to a CSV file
        descriptors_df.to_csv('Descriptors_Output.csv', index=False)
    except Exception as e:
        print(f"An error occurred: {e}")
