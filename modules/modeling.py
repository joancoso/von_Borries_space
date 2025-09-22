import os
import pickle
import numpy as np
import pandas as pd
from openTSNE.tsne import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

### Some comments:
# - Xsmall is just a smaller subset of the training data to make some steps faster
# - Caching was implemented by Sylvain. It is useful but we need to rename it.
# - The model is meant to be pickled, not really in the cache sense.
# -  I am still not so clear whether we really want/need separate preprocessing, modeling and visualization modules.


# -----------------------------
# utils? 
# -----------------------------
def ensure_dirs():
    os.makedirs("temp", exist_ok=True)
    os.makedirs("output", exist_ok=True)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------
# Preprocessing
# -----------------------------
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
# Modeling
# -----------------------------


def fit_tsne_model(Xsmall,
                   model_cache_path="temp/open_tsne_trained.pkl"):
    """
    Fits the TSNE model on the boolean fingerprint array (Xsmall) and saves the fitted embedding object.
    If the pickled object extists, it is loaded instead of training again. 
    """
    if os.path.exists(model_cache_path):
        print(f"[cache] Loading fitted TSNE embedding from {model_cache_path}")
        return load_pickle(model_cache_path)

    # @TODO: I would prefer to store all params in a config file
    tsne = TSNE(
        perplexity=100,
        n_iter=2000,
        metric='euclidean',
        random_state=42,
        verbose=True,
    )

    print('start training')
    embedding_train = tsne.fit(Xsmall)  # Try Xsmall if it crashes due to memory issues

    print("Try to pickle")
    save_pickle(embedding_train, model_cache_path)
    print(f"Saved fitted TSNE embedding to {model_cache_path}")
    return embedding_train


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


def transform_target(embedding_train,
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


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dirs()

    # 1) Load training set -> boolean array (subset) with caching
    Xsmall = load_training_space()

    # 2) Fit (or load) TSNE model
    embedding_train = fit_tsne_model(Xsmall)

    # 3) Load target dataset fingerprints (bool) with caching
    target_space_fingerprints = load_target_space()

    # 4) Transform target fingerprints into TSNE space (cache npy + csv)
    _, target_chemicals_space = transform_target(embedding_train, target_space_fingerprints)

    # 5) Plot and save figure (skip if already there)
    plot_embedding(target_chemicals_space)


if __name__ == "__main__":
    main()
