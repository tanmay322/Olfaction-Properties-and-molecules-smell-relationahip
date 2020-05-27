#!/usr/bin/python3
#
# This file is just used for training a GCN-based classifier.
#

import numpy as np
# from Utilities import *
import shap
import torch
import pickle
import sys
import timeit
import os
import Config

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cluster import spectral_clustering

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

radius = 2
dim = 50
layer =  Config.layer #5
batch = 10
lr = 1e-3
lr_decay = 0.5
decay_interval = 10
iteration = 100    # total number of epochs
extra_dim = 7+167 #20

(dim, layer, batch, decay_interval, iteration, extra_dim) = map(int, [dim, layer, batch, decay_interval, iteration, extra_dim])
lr, lr_decay = map(float, [lr, lr_decay])

dir_output = ('mydataset/classification/output/')
os.makedirs(dir_output, exist_ok=True)



dir_input = ('mydataset/classification/inputgcn_allmaccs' + str(radius) + '/')
dir_output_model = dir_output + 'model/'

dir_result = ('mydataset/classification/output/result/')
os.makedirs(dir_result, exist_ok=True)

file_result = dir_result + '.txt'

os.makedirs(dir_output_model, exist_ok=True)

fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
unknown          = 100
n_fingerprint    = len(fingerprint_dict) + unknown

file_model = dir_output_model + "model-with-layer-analysis-and-only-gcn"


from MachineLearningModels import *

dataset, feat_names, class_names = read_dataset(dir_input)
dataset_train, dataset_dev, dataset_test = construct_train_dev_test_sets(dataset, seed=1234)

feature_mask_list = [1 for j in range(dim)] + [0 for j in range(dim, dim+extra_dim)]
feature_mask_matrix = create_feature_mask_matrix(feature_mask_list)
print("dim+dim_extra = " + str(dim+extra_dim))
print("Length of feature mask list: " + str(sum(feature_mask_list)))
print(feature_mask_matrix.shape)
model = MolecularPropertyPrediction(total_dim = sum(feature_mask_list), n_fingerprint=n_fingerprint, feature_mask = feature_mask_matrix, is_multiclass=False).to(device)
trainer = Trainer(model)
tester = Tester(model)


model = do_training(iteration, trainer, tester, model, dataset_train, dataset_dev, dataset_test, file_model, file_result)




