import sys
import os
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(root, "..", "Prediction_Results.csv"))

data = df["Probability_of_Permeability"].tolist()

output_file = sys.argv[1]

with open(output_file, "w") as f:
    f.write("permeability_probability\n")
    for d in data:
        f.write(f"{d}\n")