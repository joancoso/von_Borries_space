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
def load_training_space(csv_path="input_data/kerstin_fingerprints.csv",
                        cache_path="temp/Xsmall_bool.npy",
                        use_subset=True, subset_n=6000):
    """
    Loads the von Borries training fingerprints, converts to bool,
    optionally subsets, and caches the boolean array.
    """
    if os.path.exists(cache_path):
        Xsmall = np.load(cache_path, allow_pickle=False)
        print(f"[cache] Loaded training array from {cache_path} with shape {Xsmall.shape}")
        return Xsmall

    von_borries_space = pd.read_csv(csv_path)
    print(von_borries_space.isna().sum())
    print(von_borries_space.shape)

    X = np.array(von_borries_space.astype('bool'))
    Xsmall = X[:subset_n] if use_subset else X
    print(Xsmall)

    np.save(cache_path, Xsmall, allow_pickle=False)
    print(f"[cache] Saved training array to {cache_path}")
    return Xsmall


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


def remove_chiral_centers(smiles):
    """ Support function to remove chiral information from SMILES

    Inputs
    ----------
    smiles : str, mandatory
        SMILES string

    Outputs
    ----------
    res_smiles: str
        SMILES string without chiral information

    """
    m = myMolFromSmiles(smiles)
    if m is not None:
        all_chiral = Chem.FindMolChiralCenters(m)
        all_chiral_centers = [sublist[0] for sublist in all_chiral]
        if len(all_chiral_centers) > 0:
            for each in all_chiral_centers:
                m.GetAtomWithIdx(each).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        res_smiles = Chem.MolToSmiles(m)
    else:
        res_smiles = smiles

    return res_smiles


def remove_cis_trans(smiles):
    """ Support function to remove cis/trans information from SMILES

    Inputs
    ----------
    smiles : str, mandatory
        SMILES string

    Outputs
    ----------
    res_smiles: str
        SMILES string without cis/trans information

    """
    m = myMolFromSmiles(smiles)
    if m is not None:
        for b in m.GetBonds():
            if b.GetStereo() in {Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOZ,
                                 Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS,
                                 Chem.rdchem.BondStereo.STEREOANY}:
                b.SetStereo(Chem.rdchem.BondStereo.STEREONONE)

        res_smiles = Chem.MolToSmiles(m)
    else:
        res_smiles = smiles

    return res_smiles



def create_tautomer_smiles(smiles):
    """ Function creates tautomer SMILES 

    Inputs
    ----------
    smiles : str, mandatory
        SMILES string

    Outputs
    ----------
    canonical_order_smiles : str
        canonicalized ordered SMILES string

    """

    # order SMILES
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


        canonical_order_smiles = Chem.MolToSmiles(new_mol)
        return canonical_order_smiles


def calculate_descriptors_morgan(smiles, **kwargs):
    """ Wrapper function that calculates Morgan fingerprints for a single SMILES

    Inputs
    ----------
    smiles : str, mandatory
        The SMILES string
    **kwargs: optional
        Pass in any arguments taken by rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect such as radius and nBits

    Outputs
    ----------
    array of calculated Morgan fingerprints
    """

    mol = AllChem.MolFromSmiles(smiles)
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