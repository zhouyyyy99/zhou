from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import sklearn
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from ts_transformer import import TSTransformerEncoderClassiregressor as mymodel

import eli5
from eli5.sklearn import PermutationImportance

# 1. DATASET PREPARATION
# todo: set the dataset path and data name
dataname = 'tc'
dataset_path_npz = r'/home/studio-lab-user/sagemaker-studiolab-notebooks/mydata/mdd_all_corr.npz'
dataset = np.load(dataset_path_npz)
data = dataset['tcs']
labels = dataset['labels']


# 2. MODEL SETTING
# todo: set the model path
model_path = r'/home/studio-lab-user/sagemaker-studiolab-notebooks/transformer_code/experiments/MDD_allsub_fc_train_2023-02-24_10-03-04_3l5/checkpoints/model_best.pth'
model = mymodel(feat_dim=116, max_len=116, d_model=128,
                n_heads=8,
                num_layers=3, dim_feedforward=256,
                num_classes=2,
                dropout=0.1, pos_encoding='fixed',
                activation='gelu',
                norm='BatchNorm',
                num_linear_layer=1)

start_epoch = 0
checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
state_dict = deepcopy(checkpoint['state_dict'])
# if change_output:
#     for key, val in checkpoint['state_dict'].items():
#         if key.startswith('output_layer'):
#             state_dict.pop(key)
model.load_state_dict(state_dict, strict=False)# if args.use_cuda:
#     model = model.cuda().eval()

train_X, val_X, train_y, val_y = train_test_split(data, labels, random_state=1)


perm = PermutationImportance(mymodel, val_X, val_y, random_state=1)
for i in perm.importances_mean.argsort()[::-1]:
     if perm.importances_mean[i] - 2 * perm.importances_std[i] > 0:
         print(
               f"{perm.importances_mean[i]:.3f}"
               f" +/- {perm.importances_std[i]:.3f}")
# eli5.show_weights(perm, feature_names = val_X.columns.tolist())


