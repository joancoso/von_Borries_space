import pandas as pd
from openTSNE.tsne import TSNE
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

# load the training set
von_borries_space = pd.read_csv("input_data\\kerstin_fingerprints.csv")
print(von_borries_space.isna().sum())
print(von_borries_space.shape)

hyperparameters_dict= {
    'n_components':2,
    'perplexity':100,  # default is 30
    'learning_rate':'auto',
    'early_exaggeration_iter':250,
    'early_exaggeration':'auto',
    'n_iter':2000,  # Default is 500
    'exaggeration':None,
    'dof':1,
    'theta':0.5,
    'n_interpolation_points':3,
    'min_num_intervals':50,
    'ints_in_interval':1,
    'initialization':"pca",
    'metric':"jaccard",  # deafult is euclidean
    'metric_params':None,
    'initial_momentum':0.8,
    'final_momentum':0.8,
    'max_grad_norm':None,
    'max_step_norm':5,
    'n_jobs':1,
    'neighbors':'auto',  # the default is auto
    'negative_gradient_method':'auto',
    'callbacks':None,
    'callbacks_every_iters':50,
    'random_state':None,
    'verbose':True,
    'random_state': 42,  
    }
# train the openTSNE object
# tsne = TSNE(**hyperparameters_dict)
tsne = TSNE(
    perplexity=100,
    n_iter=2000,
    metric='euclidean',
    random_state=42,
    verbose=True,
)
print('start training')
X = np.array(von_borries_space.astype('bool'))
# X = np.array(von_borries_space)
Xsmall = X[:6000]
print(Xsmall)

embedding_train = tsne.fit(Xsmall) # Try Xsmall if it crashes due to memory issues
# embedding_train = tsne.fit(X)

# Pickle it? 
print("Try to pickle")
model = embedding_train
file = open('open_tsne_trained.pkl', 'wb') # Common way to pickle by opening a file where to later dump the model
pickle.dump(model, file)
print("done!")


# Load the dataset of interest
target_space = pd.read_csv("input_data\\mfps_WWTP_combined_data.tsv", sep="\t")
target_space_fingerprints = np.array(target_space.drop(columns='CanonicalSMILES')).astype('bool')
print(target_space_fingerprints)


embedding_target_chemicals = embedding_train.transform(target_space_fingerprints)

target_chemicals_space = pd.DataFrame(embedding_target_chemicals, columns=['tsne_v1', 'tsne_v2'])

print(target_chemicals_space)
# Simple embedding for paper
ax = sns.scatterplot(data=target_chemicals_space, x='tsne_v1', y='tsne_v2',
                     s=1, alpha=1, edgecolor='black')
# ax.get_legend().remove()
ax.legend(loc='upper left', bbox_to_anchor=(1.00, 0.75), ncol=1)
# ax.set(xlim=[-200, 200], ylim=[-200, 200])
plt.axis('off')
plt.savefig('output\\target_space_static_test.tif', bbox_inches='tight', dpi=1800)
plt.close()
