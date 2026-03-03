import deepchem as dc
import pandas as pd
from rdkit import Chem

# Load ESOL (Delaney) with raw RDKit molecules
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='Raw')
train, valid, test = datasets

def dataset_to_df(dataset):
    mols = dataset.X  # RDKit Mol objects
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    y = dataset.y.flatten()
    return pd.DataFrame({"smiles": smiles, "logS": y})

df = pd.concat(
    [
        dataset_to_df(train),
        dataset_to_df(valid),
        dataset_to_df(test),
    ],
    ignore_index=True,
)

df.to_csv("data/raw/esol.csv", index=False)
print(df.head())
print("Saved to data/raw/esol.csv")