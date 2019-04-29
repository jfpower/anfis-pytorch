#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: examples showing how to use ANFIS with sklearn via skorch
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
'''


import numpy as np
from sklearn.datasets import make_classification

import torch
from torch import nn
import torch.nn.functional as F

import skorch
from skorch.callbacks import Callback

import membership
import anfis
import jang_examples
import vignette_examples
import experimental


class FittingCallback(Callback):
    '''
        In order to use ANFIS-style hybrid learning with sklearn/skorch,
        we need to add a callback to do the LSE step after each epoch.
        This class contains that callback hook.
    '''
    def __init__(self):
        super(FittingCallback, self).__init__()

    def on_epoch_end(self, net, dataset_train=None,
                     dataset_valid=None, **kwargs):
        # Get the dataset: different if we're train or train/test
        # In the latter case, we have a Subset that contains the data...
        if isinstance(dataset_train, torch.utils.data.dataset.Subset):
            dataset_train = dataset_train.dataset
        with torch.no_grad():
            net.module.fit_coeff(dataset_train.X, dataset_train.y)


# #####
# ##### First example: an ineffective FIS for a simple classifier
# ##### See: https://skorch.readthedocs.io/en/stable/user/quickstart.html
# #####


class MySimpleNet(nn.Module):
    '''
        Very simple 2-layer net, slightly adapted from the docs:
            https://skorch.readthedocs.io/en/stable/user/quickstart.html
    '''
    def __init__(self, num_in, num_feat, num_hidden=10, nonlin=F.relu):
        super(MySimpleNet, self).__init__()
        self.dense0 = nn.Linear(num_in, num_hidden)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_hidden, num_feat)
        self.output = nn.Linear(num_feat, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X))
        return X


def train_simple_nn(X, y, num_in, num_feat):
    nnet = skorch.NeuralNetClassifier(
        MySimpleNet,
        module__num_in=num_in,
        module__num_feat=num_feat,
        max_epochs=10,
        lr=0.1,
    )
    nnet.fit(X, y)
    return nnet


def fuzzy_classifier(num_in, num_mfs=5):
    '''
        Make a fuzzy classifier with 5 MFS per input, and one output
    '''
    sigma = 10 / num_mfs
    mulist = torch.linspace(-5, 5, num_mfs).tolist()
    invardefs = [('x{}'.format(i), membership.make_gauss_mfs(sigma, mulist))
                 for i in range(num_in)]
    outvars = ['y0']
    model = anfis.AnfisNet('Simple classifier', invardefs, outvars)
    return model


def train_fuzzy(model, X, y, show_plots=True):
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    net = skorch.NeuralNet(
        model,
        max_epochs=50,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.SGD,
        optimizer__lr=1e-6,
        optimizer__momentum=0.99,
        callbacks=[FittingCallback()],
    )
    if show_plots:
        experimental.plot_all_mfs(model, X)
    net.fit(X, y)
    if show_plots:
        experimental.plot_all_mfs(model, X)


def classify_example():
    num_in = 5
    num_inf = num_in - 2
    X, y = make_classification(1000, num_in,
                               n_informative=num_inf, random_state=0)
    X = X.astype(np.float32)
    model = fuzzy_classifier(num_in)
    train_fuzzy(model, X, y)
    train_simple_nn(X, y, num_in, num_inf)


# #####
# ##### Second example: Jang's example 1
# #####

def test_jang(show_plots=True):
    model = jang_examples.ex1_model()
    train_data = jang_examples.make_sinc_xy()
    X, y = train_data.dataset.tensors
    net = skorch.NeuralNet(
        model,
        max_epochs=100,
        train_split=None,
        criterion=torch.nn.MSELoss,
        #criterion__reduction='sum',
        optimizer=torch.optim.SGD,
        optimizer__lr=1e-4,
        optimizer__momentum=0.99,
        callbacks=[FittingCallback()],
    )
    net.fit(X, y)
    if show_plots:
        experimental.plot_all_mfs(model, X)
        y_actual = y
        y_pred = model(X)
        experimental.plotResults(y_actual, y_pred)


# #####
# ##### Third example: Vignette example 3
# #####

def test_vignette(show_plots=True):
    model = vignette_examples.vignette_ex3()
    X, y = jang_examples.make_sinc_xy_large().dataset.tensors
    net = skorch.NeuralNet(
        model,
        max_epochs=50,
        # train_split=None,
        # batch_size=1024,
        criterion=torch.nn.MSELoss,
        # criterion__reduction='sum',
        optimizer=torch.optim.SGD,
        optimizer__lr=1e-4,
        optimizer__momentum=0.99,
        callbacks=[FittingCallback()],
    )
    net.fit(X, y)
    if show_plots:
        experimental.plot_all_mfs(model, X)
        y_actual = y
        y_pred = model(X)
        experimental.plotResults(y_actual, y_pred)


classify_example()
#test_jang()
#test_vignette()