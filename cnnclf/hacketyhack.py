# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>


import numpy as np
import matplotlib.pyplot as plt

import theano
import lasagne

# generate the googlenet model
from revelator.google_net import build_model
#from revelator.vgg19 import build_model
cnn_layers = build_model()
cnn_input_var = cnn_layers['input'].input_var
cnn_feature_layer = cnn_layers['pool5/7x7_s1']
cnn_output_layer = cnn_layers['prob']
get_cnn_features = theano.function([cnn_input_var], lasagne.layers.get_output(cnn_feature_layer))


# download the pretrained weights
#wget -O blvc_googlenet.pkl https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/blvc_googlenet.pkl
import pickle
with open('blvc_googlenet.pkl', 'r') as f:
    model = pickle.load(f)
model_param_values = model['param values']
lasagne.layers.set_all_param_values(cnn_output_layer, model_param_values)

# load the train set
import json
from sklearn.utils import shuffle
with open('./data/meta/train.json', 'r') as f: # TODO
    train_set = json.load(f)
train_paths = train_set['scientific'] + train_set['humanity']
train_labels = np.array([1] * len(train_set['scientific']) + [0] * len(train_set['humanity']))

train_paths, train_labels = shuffle(train_paths, train_labels, random_state=42)
#train_paths = train_paths[:100]
#train_labels = train_labels[:100]
#with open('./train_path.p','wb') as f:
#    pickle.dump(train_paths,f)

# extract the label
from revelator.utils import extract_all
from revelator.google_net import prepare_image
train_features = extract_all(train_paths, './data/img', prepare_image, get_cnn_features)

# Dumping Features
with open('./data/features/train_fts.p','wb') as f:
    pickle.dump(train_features,f)
with open('./data/features/train_lbs.p','wb') as f:
    pickle.dump(train_labels,f)

# train an svm on the vgg features
from sklearn import svm, datasets
C = 10
lin_svc = svm.LinearSVC(C=C).fit(train_features, train_labels)

# load the test set
with open('./data/meta/test.json', 'r') as f:
    test_set = json.load(f)
test_paths = test_set['scientific'] + test_set['humanity']

#with open('./test_path.p','wb') as f:
#    pickle.dump(test_paths,f)

# extract the features and evaluate the svm
test_labels = np.array([1] * len(test_set['scientific']) + [0] * len(test_set['humanity']))
test_features = extract_all(test_paths, './data/img', prepare_image, get_cnn_features)

with open('./data/features/test_fts.p','wb') as f:
    pickle.dump(test_features,f)
with open('./data/features/test_lbl.p','wb') as f:
    pickle.dump(test_labels,f)

print lin_svc.score(test_features, test_labels)

