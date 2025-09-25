import pandas as pd
import numpy as np
import os
from modules.modeling import *
from modules.preprocessing import *

# load tSNE model object
reference_data_name = "data_market"
model = load_model(reference_data_name, from_zip=False)

# load data to plot
new_data_name = 'pfas_data'
new_fingerprints = preprocess_data(new_data_name)
save_fingerprints(new_fingerprints, new_data_name)
# load_fingerprints(new_data_name)

# load X
X = load_training_array(new_data_name)

# transform
coordinates = transform_target(model, X)
save_coordinates(coordinates, new_data_name)

# load coordinates directly
# coordinates = load_coordinates(new_data_name)

# plot
plot_embedding(coordinates, new_data_name)