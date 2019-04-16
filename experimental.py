#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: some simple functions to supply data and plot results.
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
"""

import itertools

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

dtype = torch.float


def sinc(x, y):
    '''
        Sinc is a simple two-input non-linear function
        used by Jang in section V of his paper (equation 30).
    '''
    def s(z):
        return (1 if z == 0 else np.sin(z) / z)
    return s(x) * s(y)


def make_sinc_xy(batch_size=1024):
    '''
        Generates a set of (x, y) values for the sync function.
        Use the range (-10,10) that was used in sec. V of Jang's paper.
    '''
    pts = torch.arange(-10, 11, 2)
    x = torch.tensor(list(itertools.product(pts, pts)), dtype=dtype)
    y = torch.tensor([[sinc(*p)] for p in x], dtype=dtype)
    td = TensorDataset(x, y)
    return DataLoader(td, batch_size=batch_size, shuffle=True)


def make_sinc_xy_large(num_cases=10000, batch_size=1024):
    '''
        Generates a set of (x, y) values for the sync function.
        Uses a large data set so we can test mini-batch in action.
    '''
    pts = torch.linspace(-10, 10, int(np.sqrt(num_cases)))
    x = torch.tensor(list(itertools.product(pts, pts)), dtype=dtype)
    y = torch.tensor([[sinc(*p)] for p in x], dtype=dtype)
    td = TensorDataset(x, y)
    return DataLoader(td, batch_size=batch_size, shuffle=True)


def make_sinc_xy2(batch_size=1024):
    '''
        A version of sinc with two outputs (sync(x) and 1-sync(x)).
    '''
    pts = list(range(-10, 11, 2))
    x = torch.tensor(list(itertools.product(pts, pts)), dtype=dtype)
    y = torch.tensor([[sinc(*p), 1-sinc(*p)] for p in x], dtype=dtype)
    td = TensorDataset(x, y)
    return DataLoader(td, batch_size=batch_size, shuffle=True)


class TwoLayerNet(torch.nn.Module):
    '''
        From the pytorch examples, a simjple 2-layer neural net.
        https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    '''
    def __init__(self, d_in, hidden_size, d_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, d_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


def linear_model(x, y, epochs=200, hidden_size=10):
    '''
        Predict y from x using a simple linear model with one hidden layer.
        https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    '''
    assert x.shape[0] == y.shape[0], 'x and y have different batch sizes'
    d_in = x.shape[1]
    d_out = y.shape[1]
    model = TwoLayerNet(d_in, hidden_size, d_out)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    errors = []
    for t in range(epochs):
        y_pred = model(x)
        tot_loss = criterion(y_pred, y)
        perc_loss = 100. * torch.sqrt(tot_loss).item() / y.sum()
        errors.append(perc_loss)
        if t % 10 == 0 or epochs < 20:
            print('epoch {:4d}: {:.5f} {:.2f}%'.format(t, tot_loss, perc_loss))
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
    return model, errors


def plotErrors(errors):
    '''
        Plot the given list of error rates against no. of epochs
    '''
    plt.plot(range(len(errors)), errors, '-ro', label='errors')
    plt.ylabel('Percentage error')
    plt.xlabel('Epoch')
    plt.show()


def plotResults(y_actual, y_predicted):
    '''
        Plot the actual and predicted y values (in different colours).
    '''
    plt.plot(range(len(y_predicted)), y_predicted.detach().numpy(),
             'r', label='trained')
    plt.plot(range(len(y_actual)), y_actual.numpy(), 'b', label='original')
    plt.legend(loc='upper left')
    plt.show()


def plotMFs(var_name, fv, x):
    '''
        A simple utility function to plot the MFs for a variable.
        Supply the variable name, MFs and a set of x values to plot.
    '''
    # Sort x so we only plot each x-value once:
    xsort, _ = x.sort()
    for mfname, yvals in fv.fuzzify(xsort):
        plt.plot(xsort.tolist(), yvals.tolist(), label=mfname)
    plt.xlabel('Values for variable {} ({} MFs)'.format(var_name, fv.num_mfs))
    plt.ylabel('Membership')
    plt.legend(bbox_to_anchor=(1., 0.95))
    plt.show()


if __name__ == '__main__':
    # Predict sinc using a simple two-layer NN, with pretty dismal results:
    x, y = make_sinc_xy().dataset.tensors
    model, errors = linear_model(x, y, 100)
    plotErrors(errors)
    plotResults(y, model(x))
