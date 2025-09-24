import pandas as pd
from openTSNE.tsne import TSNE
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from modules.modeling import *
from modules.preprocessing import *

#### files ####

# todo: do we want to add here standardization / fingerprint calculation?
# load the fingerprints of the reference space: ZeroPM marketed chemicals, version todo. for now, original fingerprints
input_df_path = os.path.join("..", "data", "data_market_tsne.csv")
fingerprints_df_path = os.path.join("..", "data", "fingerprints.csv")

# df = pd.read_csv(input_df_path)
# df['standardized SMILES'] = standardize_smiles_df(df, 'SMILES')
# df_fingerprints = pd.DataFrame(calculate_descriptors_morgan_df(df, 'standardized SMILES'))
# df_fingerprints.to_csv(fingerprints_df_path)

# fingerprints = load_fingerprints("fingerprints.csv", use_subset=True, subset_n=100)
fingerprints = load_fingerprints("fingerprints.csv")

trained_model = fit_tsne_model(fingerprints)

# #%%
# pd.DataFrame(target_space_fingerprints[0], columns=['this_works'])
# #%%
# embedding_for_plot = pd.DataFrame()
# embedding_for_plot['tsne_v1'] = embedding_test[:, 0]
# embedding_for_plot['tsne_v2'] = embedding_test[:, 1]

# plot_embedding(trained_model)

