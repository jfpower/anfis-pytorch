#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: Control examples from Jang's book, chapter 17
    Section 17.4.2: Inverse control, case study
    This is supposed to represent a 'plant' whose state is a value (y),
    and whose physical model is given by an equation (see y_next below).
    The fuzzy controller must work out the control action u.
    @author: James Power <james.power@mu.ie> May 7 2019
'''

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

import anfis
from membership import make_bell_mfs
import experimental

dtype = torch.float
np.random.seed(0)


def random_u(steps):
    '''Return a random sequence of u control values, between -1 and +1'''
    return 2 * np.random.rand(steps) - 1


def y_next(k, y, u):
    '''
        Calculate the next term in the sequence of y's (plant state).
        This is the "dynamical equation" 17.16 that defines the plant.
    '''
    if k == 0:
        return 0
    else:
        return ((y[k-1] * u[k-1]) / (1 + y[k-1] ** 2)) - np.tan(u[k-1])


def make_plant_seq(steps, u):
    ''' Return a list of y terms that represents the plant'''
    y = []
    for k in range(steps):
        y.append(y_next(k, y, u))
    return y


def get_training(size=100, and_plot=True):
    '''Return three lists representing y(k), y(k+1), u(k) for k in 0..size-1'''
    u = random_u(size)
    y = make_plant_seq(size+1, u)
    y_now = y[:-1]
    y_next = y[1:]
    if and_plot:
        plt.scatter(y_now, y_next, marker='o', label='Training Data')
        plt.xlabel('y_now: y value at time t')
        plt.ylabel('y_next: y value at time t+1')
        plt.title('Training data ({} points)'.format(size))
        plt.show()
    return (y_now, y_next, u)


def make_training_data(size=100):
    '''
        Create the training data and wrap it in a torch DataLoader
    '''
    y_now, y_next, u = get_training(size)
    train_inp = torch.tensor([y_now, y_next], dtype=dtype).t()
    train_out = torch.tensor(u, dtype=dtype).reshape(size, 1)
    td = TensorDataset(train_inp, train_out)
    return DataLoader(td, batch_size=size, shuffle=False)


def plant_model_untrained():
    '''
        Thhis is the initial (untrained) model: (y_now, y_next) -> u
        Uses 3 Bell membership functions for each input.
    '''
    invardefs = [
            ('y_now', make_bell_mfs(1, 2, [-1, 0, 1])),
            ('y_next', make_bell_mfs(1, 2, [-1, 0, 1])),
            ]
    outvars = ['u']
    anf = anfis.AnfisNet('Plant Model', invardefs, outvars)
    return anf


# ###
# ### Testing
# ###

def u_next(model, y_now, y_next):
    '''
        Use the model to predict the next u value, given a y_t, y_t+1 pair.
        Returns a single scalar value.
    '''
    test_in = torch.tensor([y_now, y_next], dtype=dtype).reshape(1, 2)
    return model(test_in).item()


def run_plant_trained(steps, model, y_desired, y_init=0.0):
    '''
        Run the model for steps, using y_desired as the intended trajectory.
        That is, use the model and y_desired to get a u value,
            then use this u-value to calculate the next (actual) y value.
        Return two lists: the actual y values and the control actions.
    '''
    y_act = [y_init]  # These are the 'real' y-values, via the plant equation
    u = []  # These are the control actions, predicted by the model.
    for k in range(steps-1):
        u.append(u_next(model, y_act[k], y_desired[k+1]))
        y_act.append(y_next(k, y_act, u))
    return (y_act, u)


def make_y_desired(size=100):
    '''
        The desired trajectory of y-values,
        returned as a list of the given size.
    '''
    return [0.6 * np.sin(2*np.pi*k/250) + 0.2*np.sin(2*np.pi*k/50)
            for k in range(size)]


def test_control_model(model, size=300, and_plot=True):
    print('### Testing, dataset size = {} cases'.format(size))
    y_desired = make_y_desired(size)
    y_act, u = run_plant_trained(size, model, y_desired)
    if and_plot:
        plt.plot(range(len(y_act)), y_act, 'g', label='trained')
        plt.plot(range(len(y_desired)), y_desired, 'b', label='desired')
        plt.xlabel('Time')
        plt.ylabel('Trained and Desired y(t)')
        plt.legend(loc='upper right')
        plt.show()
        error = [a-d for (d, a) in zip(y_desired, y_act)]
        plt.plot(range(len(error)), error, 'r', label='error')
        plt.xlabel('Time')
        plt.ylabel('Error in y(t)')
        plt.legend(loc='upper right')
        plt.hlines(y=0, xmin=0, xmax=size, linestyle=':', color='grey')
        plt.show()
    mse, rmse, perc_loss = experimental.calc_error(
            torch.tensor(y_act), torch.tensor(y_desired))
    print('On {} test cases, MSE={:.5f}, RMSE={:.5f} ={:.2f}%'
          .format(size, mse, rmse, perc_loss))
    #model.layer.fuzzify.show()
    return (y_act, u)



model = plant_model_untrained()

print('### TRAINING ###')
train_data = make_training_data()
experimental.plot_all_mfs(model, train_data.dataset.tensors[0])
optimizer = torch.optim.Rprop(model.parameters(), lr=1e-3, etas=(0.9, 1.2))
criterion = torch.nn.MSELoss(reduction='sum')
experimental.train_anfis_with(model, train_data, optimizer, criterion,
                              epochs=150, show_plots=True)
experimental.plot_all_mfs(model, train_data.dataset.tensors[0])
print('### TESTING ###')
test_control_model(model)
