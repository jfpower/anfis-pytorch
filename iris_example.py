#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: test cases form the Vignette paper
        This is example 5, the "Iris data" - I'm using the sklearn data.
        "ANFIS vignette" by Cristobal Fresno and Elmer A. Fernández,
        http://www.bdmg.com.ar/?page_id=176, or CRAN package 'anfis'
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
'''

import sklearn.datasets

import torch
from torch.utils.data import TensorDataset, DataLoader

import anfis
from membership import make_gauss_mfs, make_anfis
import experimental


def make_one_hot(data, num_categories, dtype=torch.float):
    '''
        Take a list of categories and make them into one-hot vectors;
        that is, treat the original entries as vector indices.
        Return a tensor of 0/1 floats, of shape: len(data) * num_categories
    '''
    num_entries = len(data)
    # Convert data to a torch tensor of indices, with extra dimension:
    cats = torch.Tensor(data).long().unsqueeze(1)
    # Now convert this to one-hot representation:
    y = torch.zeros((num_entries, num_categories), dtype=dtype)\
        .scatter(1, cats, 1)
    y.requires_grad = True
    return y


def get_iris_data_one_hot(in_feat=2, batch_size=1024):
    '''
        Return the iris data as a torch DataLoader object.
        There are 4 input features, but you can select fewer.
        The y-values are a one-hot representation of the categories.
    '''
    d = sklearn.datasets.load_iris()
    x = torch.Tensor(d.data[:, :in_feat])
    y = make_one_hot(d.target, num_categories=3)
    td = TensorDataset(x, y)
    return DataLoader(td, batch_size=batch_size, shuffle=False)


def get_iris_data(in_feat=2, batch_size=1024):
    '''
        Return the iris data as a torch DataLoader object.
        There are 4 input features, but you can select fewer.
        The y values are just the number indicating the category.
    '''
    d = sklearn.datasets.load_iris()
    x = torch.Tensor(d.data[:, :in_feat])
    y = torch.Tensor(d.target).unsqueeze(1)
    td = TensorDataset(x, y)
    return DataLoader(td, batch_size=batch_size, shuffle=False)


def vignette_ex5(in_feat=2):
    '''
        These are the original (untrained) MFS for Vignette example 5.
        Uses 3 Gaussian membership functions for each of 3 inputs
        (but the Iris data has 4 inputs, so I'll use 4 variables)
    '''
    def mk_var(name):
        return (name, make_gauss_mfs(1, [-5, 0.08, 1.67]))
    invardefs = [mk_var(name) for name in
                 ['sepal length (cm)',  'sepal width (cm)',
                  'petal length (cm)', 'petal width (cm)'][:in_feat]
                 ]
    outvars = ['setosa', 'versicolor', 'virginica']
    anf = anfis.AnfisNet('Iris Plants Database', invardefs, outvars)
    return anf


def num_cat_correct(model, x, y_actual):
    '''
        Work out the number of correct categorisations the model gives.
        Assumes the model is producing (float) scores for each category.
        Use a max function on predicted/actual to get the category.
    '''
    y_pred = model(x)
    # Change the y-value scores back into 'best category':
    cat_act = torch.argmax(y_actual, dim=1)
    cat_pred = torch.argmax(y_pred, dim=1)
    num_correct = torch.sum(cat_act == cat_pred)
    return num_correct.item(), len(x)


def train_hybrid(in_feat=2):
    '''
        Train a hybrid Anfis based on the Iris data.
        I use a 'resilient' BP optimiser here, as SGD was a little flakey.
    '''
    train_data = get_iris_data_one_hot(in_feat)
    x, y_actual = train_data.dataset.tensors
    model = make_anfis(x, num_mfs=3, num_out=3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
    optimizer = torch.optim.Rprop(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss(reduction='sum')
    experimental.train_anfis_with(model, train_data, optimizer, criterion, 5)
    experimental.plot_all_mfs(model, x)
    nc, tot = num_cat_correct(model, x, y_actual)
    print('{} of {} correct (={:5.2f}%)'.format(nc, tot, nc*100/tot))
    return model


def train_non_hybrid(in_feat=2):
    '''
        Train a non-hybrid Anfis for the Iris data (so, no LSE).
        Loss criterion is CrossEntropy, and expects target to be categories.
        Note that the model still produces (float) scores for each category.
    '''
    train_data = get_iris_data(in_feat)
    x, y_actual = train_data.dataset.tensors
    model = make_anfis(x, num_mfs=3, num_out=3, hybrid=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)
    def criterion(input, target):  # change the dim and type
        return torch.nn.CrossEntropyLoss()(input, target.squeeze().long())
    experimental.train_anfis_with(model, train_data, optimizer, criterion, 250)
    y_pred = model(x)
    nc = torch.sum(y_actual.squeeze().long() == torch.argmax(y_pred, dim=1))
    tot = len(x)
    experimental.plot_all_mfs(model, x)
    print('{} of {} correct (={:5.2f}%)'.format(nc, tot, nc*100/tot))
    return model


model = train_non_hybrid()
# print(model.coeff)
