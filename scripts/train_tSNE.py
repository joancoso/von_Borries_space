import pandas as pd
import numpy as np
import os
from modules.modeling import *
from modules.preprocessing import *

#### input ####
input_data_name = "data_market" # file "data_market.csv" should be saved under "data"

#### preprocessing and calculating fingerprints ####
# df_fingerprints = preprocess_data(input_data_name)
# save_fingerprints(df_fingerprints, input_data_name)
df_fingerprints = load_fingerprints(input_data_name)

#### load training_array from fingerprints file ####
# --> can be used if fingerprints are already calculated and available in output folder
# training_array = load_training_array(input_data_name,
#                                       use_subset=True,
#                                       subset_n=100) # subset for testing
training_array = load_training_array(input_data_name) # full set

#### train model ####
trained_model, coordinates = fit_tsne_model(training_array)
save_model(trained_model, input_data_name)
save_coordinates(coordinates, input_data_name, df_fingerprints['INCHIKEY'])

#### plot tsne #### # todo ---> this should go somewhere else I think

# pd.DataFrame(target_space_fingerprints[0], columns=['this_works'])
#
# embedding_for_plot = pd.DataFrame()
# embedding_for_plot['tsne_v1'] = embedding_test[:, 0]
# embedding_for_plot['tsne_v2'] = embedding_test[:, 1]
#
# plot_embedding(trained_model)

