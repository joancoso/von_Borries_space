import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize



# -----------------------------
# Data and model loading
# -----------------------------
# Do we really need this function actually? It is just a few lines of code.
def load_fingerprints(fingerprints_file="kerstin_fingerprints.csv",
                        cache_path="X_bool.npy",
                        use_subset=False, subset_n=6000):
    """
    Loads the von Borries training fingerprints, converts to bool,
    optionally subsets, and caches the boolean array.
    """
    # Create os-independent file paths
    fingerprints_file = os.path.join("..", "data", fingerprints_file)
    cache_path = os.path.join("..", "data", cache_path)

    if os.path.exists(cache_path):
        X = np.load(cache_path, allow_pickle=False)
        print(f"[cache] Loaded training array from {cache_path} with shape {X.shape}")
        return X

    # load fingerprints
    fingerprints = pd.read_csv(fingerprints_file)

    # ensure that there is not NA in the data
    assert fingerprints.isna().sum().sum() == 0, "There are NaNs in the data"
    print("Loaded fingerprints data size:", fingerprints.shape)

    # obtain training data
    X = np.array(fingerprints).astype('bool')  # full data
    X = X[:subset_n] if use_subset else X
    print("Size of X:", X.size)

    np.save(cache_path, X, allow_pickle=False)
    print(f"[cache] Saved training array to {cache_path}")
    return X

# -----------------------------
# Structures and features
# -----------------------------

def myMolFromSmiles(smiles):
    """ Function to create mol object from SMILES performing partial sanitization when necessary

    Inputs
    ----------
    smiles : str, mandatory
        SMILES string

    Outputs
    ----------
    mol: object
        RDKit mol object

    """
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # try partial sanitization
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol,
                             Chem.SanitizeFlags.SANITIZE_FINDRADICALS | Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                             Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                             Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                             catchErrors=True)
            print('Partial sanitization: ' + smiles)
        except:
            print('Partial sanitization failed - return none: ' + smiles)

    return mol


def remove_stereochemistry(smiles):
    """ Support function to remove stereochemical information from SMILES

    Inputs
    ----------
    smiles : str, mandatory
        SMILES string

    Outputs
    ----------
    res_smiles: str
        SMILES string without stereochemical information

    """

    mol = myMolFromSmiles(smiles)

    if mol is None:
        res_smiles = ""
    else:
        Chem.RemoveStereochemistry(mol) 
        res_smiles = Chem.MolToSmiles(mol)

    return res_smiles


def create_tautomer_smiles(smiles):
    """ Function creates tautomer SMILES 

    Inputs
    ----------
    smiles : str, mandatory
        SMILES string

    Outputs
    ----------
    tautomer_smiles : str
        tautomer SMILES string

    """

    if myMolFromSmiles(smiles) is None:
        new_mol = None
    else:
        mod_smi = Chem.MolToSmiles(myMolFromSmiles(smiles))
        new_mol = myMolFromSmiles(mod_smi)

    if new_mol is None:  # return empty string if still no mol
        print('No mol: ' + smiles)
        return ''
    else:
        # Tautomerize
        try:
            enumerator = rdMolStandardize.TautomerEnumerator()
            new_mol = enumerator.Canonicalize(new_mol)
        except:
            print('No tautomerization:' + smiles)


        tautomer_smiles = Chem.MolToSmiles(new_mol)
        return tautomer_smiles


def standardize_smiles(smiles, tautomerize=False):
    """ Wrapper function that standardizes SMILES by removing stereochemistry and optional tautomerizing 
    Inputs
    ----------
    smiles : str, mandatory
        The SMILES string

    Outputs
    ----------
    smiles_std: str,
        The standardized SMILES string
    """ 
    if tautomerize:
        smiles = create_tautomer_smiles(smiles)

    smiles_std = remove_stereochemistry(smiles)

    return smiles_std


def standardize_smiles_df(df, col_smiles, **kwargs):
    """ Wrapper function that calculates Morgan fingerprints for a series of SMILES

    Inputs
    ----------
    df : pandas dataframe, mandatory
        The dataframe containing the series of SMILES
    col_smiles: string, mandatory
        The column name containing the SMILES
    **kwargs: optional
        Pass in any arguments taken by rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect such as radius and nBits

    Outputs
    ----------
    series of standardized SMILES
    """

    smiles_df = df[col_smiles].apply(standardize_smiles, **kwargs)
    return smiles_df


def calculate_descriptors_morgan(smiles, **kwargs):
    """ Wrapper function that calculates Morgan fingerprints for a single SMILES

    Inputs
    ----------
    smiles : str, mandatory
        The SMILES string
    **kwargs: optional
        Pass in any arguments taken by rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect such as radius and fpSize

    Outputs
    ----------
    array of calculated Morgan fingerprints
    """

    mol = myMolFromSmiles(smiles)
    gen = rdFingerprintGenerator.GetMorganGenerator(**kwargs)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=bool)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


def calculate_descriptors_morgan_df(df, col_smiles, **kwargs):
    """ Wrapper function that calculates Morgan fingerprints for a series of SMILES

    Inputs
    ----------
    df : pandas dataframe, mandatory
        The dataframe containing the series of SMILES
    col_smiles: string, mandatory
        The column name containing the SMILES
    **kwargs: optional
        Pass in any arguments taken by rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect such as radius and nBits

    Outputs
    ----------
    dataframe of calculated Morgan fingerprints
    """

    d = df[col_smiles].apply(calculate_descriptors_morgan, **kwargs)
    return pd.DataFrame.from_records(d)