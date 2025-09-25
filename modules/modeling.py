import os
import pickle
import numpy as np
import pandas as pd
from openTSNE.tsne import TSNE
# from openTSNE.sklearn import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from modules.preprocessing import *
from zipfile import ZipFile
import yaml


### Some comments:
# - Xsmall is just a smaller subset of the training data to make some steps faster
# - Caching was implemented by Sylvain. It is useful but we need to rename it.
# - The model is meant to be pickled, not really in the cache sense.
# -  I am still not so clear whether we really want/need separate preprocessing, modeling and visualization modules.


# -----------------------------
# utils? 
# -----------------------------
def ensure_dirs():
    """
    Ensure that the folders temp and output exist, and if not, create them
    """
    os.makedirs(os.path.join("..", "temp"), exist_ok=True)
    os.makedirs(os.path.join("..", "output"), exist_ok=True)


def save_pickle(obj, path):
    """
    Save object as a pickle file
    :param obj: objeckt
    :param path: path to save pickle file
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """
    Load pickled file
    :param path: path to saved pickle file
    :return: object in the pickled file
    """
    with open(path, "rb") as f:
        return pickle.load(f)
    
def load_configs(params = 'tsne_params'):
    """
    Load tSNE configurations from the config folder
    :param params: name of file in config folder (without the .yaml)
    :return: dictionary of hyperparameters to be used for tSNE training
    """
    with open(os.path.join('..', 'config', params+'.yaml'), 'r') as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters['hyperparameters']

# -----------------------------
# Modeling
# -----------------------------


def fit_tsne_model(X):
    """
    Fits the TSNE model on the boolean fingerprint array (X) and saves the fitted embedding object.
    If the pickled object exists, it is loaded instead of training again.

    :param X: input matrix of fingerprints
    :return: fitted model object and coordinates of transformed fingerprints
    """

    # Load hyperparameters of t-SNE from config
    hyperparameters_dict = load_configs('tsne_params')

    # Training
    print('--> Start training')
    tsne = TSNE(**hyperparameters_dict)
    model = tsne.fit(X)
    coordinates = pd.DataFrame(model.transform(X))
    # coordinates, model = tsne.fit_transform(X)
    print('Finished training')
    return model, coordinates

def save_model(model, filename):
    """
    Save model as pickle to temp and as zipped pickle to output
    @param model: trained tSNE model
    @param filename: name tag of the original data file (e.g. for 'data_market.csv' the filename would be 'data_market')
    """
    ensure_dirs()
    model_path = os.path.join("..", "temp", filename + '_trained_tSNE.pkl')
    model_path_zip = os.path.join("..", "output", filename + '_trained_tSNE.zip')

    # Saving trained tSNE object to temp folder
    print("--> Pickle tSNE object")
    save_pickle(model, model_path)
    print(f"Saved fitted tSNE embedding to {model_path}")

    # Saving zipped trained tSNE object to output folder
    print('--> Zip tSNE object')
    with ZipFile(model_path_zip, "w") as zipf:
        zipf.write(model_path)
    print(f"Saved fitted tSNE embedding as zip file to {model_path_zip}")

def save_coordinates(coordinates, filename, inchikeys = None):
    """
    Save coordinates to csv file with the columns TSNE1 and TSNE2

     #todo: check for consistency - ideally we would alsways get the same output here for visualization

    :param coordinates: coordinates as received from model fitting
    :param filename: name tag of the original data file (e.g. for 'data_market.csv')
    :param inchikeys: optional - list of inchikeys, for example from original data file
    """
    coordinates_path = os.path.join("..", "temp", filename + '_coordinates_tSNE.csv')
    # coordinates.columns = ['TSNE1', 'TSNE2']
    if inchikeys:
        coordinates.index = inchikeys
    coordinates.to_csv(coordinates_path, index=True)

def load_coordinates(filename):
    coordinates_path = os.path.join("..", "temp", filename + '_coordinates_tSNE.csv')
    coordinates = pd.read_csv(coordinates_path)
    return coordinates

def load_model(filename, from_zip = False):
    """
    Load model from pickle file (default) or from zip file (not implemented)
    :param filename: name tag of the original data file (e.g. 'data_market' for 'data_market.csv')
    :param from_zip: Load from zip file #todo implement this option
    :return: model object
    """
    if from_zip:
        model_path_zip = os.path.join("..", "output", filename + '_trained_tSNE.zip')
        model_name = os.path.join("..", "temp", filename + '_trained_tSNE.pkl')
        archive = ZipFile(model_path_zip, 'r')
        model = archive.read(model_name) # todo: this does not work - it says it's a possible zip bomb (:
    else:
        model_path = os.path.join("..", "temp", filename + '_trained_tSNE.pkl')
        model = load_pickle(model_path)
    return model

# -----------------------------
# Modeling (for the target space)
# -----------------------------
def load_target_space(tsv_path="data/target_example_data.tsv",
                      fps_cache_path="temp/target_example_fingerprints_bool.npy"):
    """
    Loads target dataset, extracts fingerprints (drop CanonicalSMILES), converts to bool, caches the matrix.
    """

    # I wonder if the cache here is really necessary, as this is a small file anyway
    if os.path.exists(fps_cache_path):
        target_space_fingerprints = np.load(fps_cache_path, allow_pickle=False)
        print(f"[cache] Loaded target fingerprints from {fps_cache_path} with shape {target_space_fingerprints.shape}")
        return target_space_fingerprints

    target_space = pd.read_csv(tsv_path, sep="\t")
    target_space_fingerprints = np.array(target_space.drop(columns='CanonicalSMILES')).astype('bool')
    print(target_space_fingerprints)

    np.save(fps_cache_path, target_space_fingerprints, allow_pickle=False)
    print(f"[cache] Saved target fingerprints to {fps_cache_path}")
    return target_space_fingerprints


def transform_target(embedding_train,           # todo: code from José - I simplified it below, what do you think?
                     target_space_fingerprints,
                     emb_cache_path="temp/embedding_target_chemicals.npy",
                     df_cache_path="temp/target_chemicals_space.csv"):
    """
    Transforms target fingerprints into the trained TSNE space, caches both the raw embedding and a CSV.
    """
    if os.path.exists(emb_cache_path) and os.path.exists(df_cache_path):
        print(f"[cache] Loading transformed embedding from {emb_cache_path} and {df_cache_path}")
        embedding_target_chemicals = np.load(emb_cache_path, allow_pickle=False)
        target_chemicals_space = pd.read_csv(df_cache_path)
        return embedding_target_chemicals, target_chemicals_space

    embedding_target_chemicals = embedding_train.transform(target_space_fingerprints)
    target_chemicals_space = pd.DataFrame(embedding_target_chemicals, columns=['tsne_v1', 'tsne_v2'])

    np.save(emb_cache_path, embedding_target_chemicals, allow_pickle=False)
    target_chemicals_space.to_csv(df_cache_path, index=False)
    print(f"[cache] Saved transformed embedding to {emb_cache_path}")
    print(f"[cache] Saved target chemicals space CSV to {df_cache_path}")

    print(target_chemicals_space)
    return embedding_target_chemicals, target_chemicals_space

def transform_target(model, target_X):
    coordinates_target = model.transform(target_X)
    coordinates_df = pd.DataFrame(coordinates_target, columns=['TSNE1', 'TSNE2'])
    return coordinates_df

# -----------------------------
# Visualization
# -----------------------------
def plot_embedding(target_chemicals_space,
                   fig_path='output/target_space_static_test.tif'):
    """
    Plots the simple scatter as in the original script and saves it.
    Skips re-plot if the file already exists.
    """
    if os.path.exists(fig_path):
        print(f"[cache] Figure already exists at {fig_path}; skipping re-plot.")
        return

    ax = sns.scatterplot(data=target_chemicals_space, x='tsne_v1', y='tsne_v2',
                         s=1, alpha=1, edgecolor='black')
    ax.legend(loc='upper left', bbox_to_anchor=(1.00, 0.75), ncol=1)
    plt.axis('off')
    plt.savefig(fig_path, bbox_inches='tight', dpi=1800)
    plt.close()
    print(f"[out] Saved figure to {fig_path}")

def plot_embedding(coordinates, filename, format = '.tif'):
    """
    Plot a coordinates file #todo define what we really need here
    :param coordinates: coordinates dataframe
    :param filename: name tag
    :param format: output format of the plot (e.g., '.png', '.pdf'). '.tif' by default
    """
    print(f"--> Plotting {filename}")
    ax = sns.scatterplot(data=coordinates, x='TSNE1', y='TSNE2',
                         s=1, alpha=1, edgecolor='black')
    fig_path = os.path.join("..", "output", filename + "plot" + format)
    ax.legend(loc='upper left', bbox_to_anchor=(1.00, 0.75), ncol=1)
    plt.axis('off')
    plt.savefig(fig_path, bbox_inches='tight', dpi=1800)
    plt.close()
    print(f"[out] Saved figure to {fig_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dirs()

    # 1) Load training set -> boolean array (subset) with caching
    X = load_training_array()

    # 2) Fit (or load) TSNE model
    embedding_train, _ = fit_tsne_model(X) # _ catches the coordinates

    # 3) Load target dataset fingerprints (bool) with caching
    target_space_fingerprints = load_target_space()

    # 4) Transform target fingerprints into TSNE space (cache npy + csv)
    _, target_chemicals_space = transform_target(embedding_train, target_space_fingerprints)

    # 5) Plot and save figure (skip if already there)
    plot_embedding(target_chemicals_space)


if __name__ == "__main__":
    main()
