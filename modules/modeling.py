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
# Modeling
# -----------------------------


def fit_tsne_model(X, model_cache_path="open_tsne_trained.pkl"):
    """
    Fits the TSNE model on the boolean fingerprint array (X) and saves the fitted embedding object.
    If the pickled object extists, it is loaded instead of training again.
    """
    model_cache_path = os.path.join("..", "output", model_cache_path)

    # @TODO: I would prefer to store all params in a config file
    # Define hyperparameters of t-SNE
    hyperparameters_dict = {
        'n_components': 2,
        'perplexity': 100,  # default is 30
        'learning_rate': 'auto',
        'early_exaggeration_iter': 250,
        'early_exaggeration': 'auto',
        'n_iter': 2000,  # Default is 500
        'exaggeration': None,
        'dof': 1,
        'theta': 0.5,
        'n_interpolation_points': 3,
        'min_num_intervals': 50,
        'ints_in_interval': 1,
        'initialization': "pca",
        'metric': "jaccard",  # deafult is euclidean
        'metric_params': None,
        'initial_momentum': 0.8,
        'final_momentum': 0.8,
        'max_grad_norm': None,
        'max_step_norm': 5,
        'n_jobs': 1,
        'neighbors': 'auto',  # the default is auto
        'negative_gradient_method': 'auto',
        'callbacks': None,
        'callbacks_every_iters': 50,
        'random_state': None,
        'verbose': True,
        'random_state': 42,
    }

    # These are the settings I usually use (Kerstin) - parameters we might want to review in particular: perplexity, n_iter (both during fitting and transforming)
    # tsne = TSNE(n_components=2, perplexity=100, n_iter=2000, learning_rate='auto', neighbors='pynndescent',
    #         initialization='pca', metric='jaccard', random_state=42, verbose=3)

    # Training
    print('start training')
    tsne = TSNE(**hyperparameters_dict)
    embedding_train = tsne.fit(X)  # Try Xsmall if it crashes due to memory issues
    print('finished training')

    # Saving trained tSNE object
    print("Try to pickle")
    save_pickle(embedding_train, model_cache_path)
    print(f"Saved fitted tSNE embedding to {model_cache_path}")
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
    X = load_fingerprints()

    # 2) Fit (or load) TSNE model
    embedding_train = fit_tsne_model(X)

    # 3) Load target dataset fingerprints (bool) with caching
    target_space_fingerprints = load_target_space()

    # 4) Transform target fingerprints into TSNE space (cache npy + csv)
    _, target_chemicals_space = transform_target(embedding_train, target_space_fingerprints)

    # 5) Plot and save figure (skip if already there)
    plot_embedding(target_chemicals_space)


if __name__ == "__main__":
    main()
