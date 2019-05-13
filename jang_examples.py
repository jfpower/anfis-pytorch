#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: Examples from Jang's paper
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
'''

import sys
import itertools
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

import anfis
from membership import BellMembFunc, make_bell_mfs
from experimental import train_anfis, test_anfis

dtype = torch.float


# ##### Example 1: Modeling a Two-Input Nonlinear Function #####


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
        Not part of Jang's work, but used by the Vignette paper.
    '''
    pts = list(range(-10, 11, 2))
    x = torch.tensor(list(itertools.product(pts, pts)), dtype=dtype)
    y = torch.tensor([[sinc(*p), 1-sinc(*p)] for p in x], dtype=dtype)
    td = TensorDataset(x, y)
    return DataLoader(td, batch_size=batch_size, shuffle=True)


def ex1_model():
    '''
        These are the original (untrained) MFS for Jang's example 1.
    '''
    invardefs = [
            ('x0', make_bell_mfs(3.33333, 2, [-10, -3.333333, 3.333333, 10])),
            ('x1', make_bell_mfs(3.33333, 2, [-10, -3.333333, 3.333333, 10])),
            ]
    outvars = ['y0']
    anf = anfis.AnfisNet('Jang\'s example 1', invardefs, outvars)
    return anf


# ##### Example 2: Modeling a Three-Input Nonlinear Function #####

def ex2_eqn(x, y, z):
    '''
        The three input non-linear function used in Jang's example 2
    '''
    output = 1 + torch.pow(x, 0.5) + torch.pow(y, -1) + torch.pow(z, -1.5)
    output = torch.pow(output, 2)
    return output


def _make_data_xyz(inp_range):
    '''
        Given a range, return a dataset with the product of these values.
        Assume we want triples returned - i.e. (x,y,z) points
    '''
    xyz_vals = itertools.product(inp_range, inp_range, inp_range)
    x = torch.tensor(list(xyz_vals), dtype=dtype)
    y = torch.tensor([[ex2_eqn(*p)] for p in x], dtype=dtype)
    return TensorDataset(x, y)


def ex2_model():
    invardefs = [
            ('x', make_bell_mfs(2.5, 2, [1, 6])),
            ('y', make_bell_mfs(2.5, 2, [1, 6])),
            ('z', make_bell_mfs(2.5, 2, [1, 6])),
            ]
    outvars = ['output']
    model = anfis.AnfisNet('Jang\'s example 2', invardefs, outvars)
    return model


def ex2_training_data(batch_size=1024):
    '''
        Jang's training data uses integer values between 1 and 6 inclusive
    '''
    inp_range = range(1, 7, 1)
    td = _make_data_xyz(inp_range)
    return DataLoader(td, batch_size=batch_size, shuffle=True)


def ex2_testing_data():
    '''
        Jang's test data uses values 1.5, 2.5 etc.
    '''
    inp_range = np.arange(1.5, 6.5, 1)
    td = _make_data_xyz(inp_range)
    return DataLoader(td)


# ##### Example 3: On-line Identification in Control Systems #####


def ex3_model(mfnum=7):
    '''
        Example 3 model, with variable number of Bell MFs, range (-1,+1).
        Specify the no. of MFs, or make it 0 and I'll use Jang's 5 centers.
        Either way, the Bell width/slope values are from Jang's data.
    '''
    # The paper says 7 MFs are best, but his code uses 5 MFs
    if mfnum < 1:  # use the 5 MF values from Jang's code
        centers = [-0.999921, -0.499961, 0.000000, 0.499961, 0.99992]
    else:  # just spread them evenly accross the range (-1, +1)
        centers = np.linspace(-1, 1, mfnum)
    invardefs = [('k', make_bell_mfs(0.249980, 4, centers))]
    outvars = ['y']
    model = anfis.AnfisNet('Jang\'s example 3', invardefs, outvars)
    return model


def ex3_f(u):
    '''
        This is the function f defined in eq 34 of Jang's paper.
        This is the function that the ANFIS is supposed to model.
    '''
    pi_u = np.pi * u
    return (0.6 * torch.sin(pi_u) +
            0.3 * torch.sin(3 * pi_u) +
            0.1 * torch.sin(5 * pi_u))


def ex3_training_data(batch_size=1024):
    '''
        Jang's training data, spread evenly over -1 to +1
    '''
    inp_range = np.arange(-1, 1.02, 0.02)
    # Need to add an extra dimension to both x and y:
    x = torch.tensor(inp_range, dtype=dtype).unsqueeze(1)
    y = ex3_f(x)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def ex3_u(k):
    '''
        This is the input function u(k) defined in eq 33 of Jang's paper.
        The purpose of this function is to generate (test) input values.
        Note that this is a scalar -> scalar function (no tensors).
        For positive integer argument, return a float in the range -1 to +1.
    '''
    assert k >= 1, 'not defined for k={}, only 1 or over'.format(k)
    if k < 500:
        u = np.sin((2 * np.pi * k) / 250)
    else:  # Over 500, use a different formula:
        u = 0.5 * np.sin((2 * np.pi * k) / 250) + \
            0.5 * np.sin((2 * np.pi * k) / 25)
    return u


def ex3_testing_data():
    '''
        As test data, use the valus generated by the u(k) function
    '''
    x = torch.tensor([ex3_u(k) for k in range(1, 700)]).unsqueeze(1)
    y = ex3_f(x)
    return DataLoader(TensorDataset(x, y))


# ##### Example 4: predicting Chaotic Dynamics #####

def ex4_model():
    '''
        Example 4 model, from Jang's data; 4 variables with 2 MFs each.
        Predict x(t+6) based on x(t-18), x(t-12), x(t-6), x(t)
        These are the starting MFs values he suggests.
    '''
    invardefs = [
            ('xm18', make_bell_mfs(0.444045, 2, [0.425606, 1.313696])),
            ('xm12', make_bell_mfs(0.444045, 2, [0.425606, 1.313696])),
            ('xm6',  make_bell_mfs(0.444045, 2, [0.425606, 1.313696])),
            ('x',    make_bell_mfs(0.444045, 2, [0.425606, 1.313696])),
            ]
    outvars = ['xp6']
    model = anfis.AnfisNet('Jang\'s example 4', invardefs, outvars)
    return model


def jang_ex4_trained_model():
    '''
        Example 4 model, from Jang's data; 4 variables with 2 MFs each.
        These are the final 'trained' values from pg. 683.
    '''
    # Data from Table VI:
    mfs = [
        (0.1790, 2.0456, 0.4798),  # SMALL1
        (0.1584, 2.0103, 1.4975),  # LARGE1
        (0.2410, 1.9533, 0.2960),  # SMALL2
        (0.2923, 1.9178, 1.7824),  # LARGE2
        (0.3798, 2.1490, 0.6599),  # SMALL3
        (0.4884, 1.8967, 1.6465),  # LARGE3
        (0.2815, 2.0170, 0.3341),  # SMALL4
        (0.1616, 2.0165, 1.4727),  # LARGE4
    ]
    invardefs = [
            ('xm18', [BellMembFunc(*mfs[0]), BellMembFunc(*mfs[1])]),
            ('xm12', [BellMembFunc(*mfs[2]), BellMembFunc(*mfs[3])]),
            ('xm6',  [BellMembFunc(*mfs[4]), BellMembFunc(*mfs[5])]),
            ('x',    [BellMembFunc(*mfs[6]), BellMembFunc(*mfs[7])]),
            ]
    outvars = ['xp6']
    model = anfis.AnfisNet('Jang\'s example 4 (trained)', invardefs, outvars)
    # Jang calls this "the parameter matrix C" on pg 683:
    coeff = torch.tensor([
        [0.2167,   0.7233, -0.0365,  0.5433,  0.0276],
        [0.2141,   0.5704, -0.4826,  1.2452, -0.3778],
        [-0.0683,  0.0022,  0.6495,  2.7320, -2.2916],
        [-0.2616,  0.9190, -2.9931,  1.9467,  1.6555],
        [-0.3293, -0.8943,  1.4290, -1.6550,  2.3735],
        [2.5820,  -2.3109,  3.7925, -5.8068,  4.0478],
        [0.8797,  -0.9407,  2.2487,  0.7759, -2.0714],
        [-0.8417, -1.5394, -1.5329,  2.2834,  2.4140],
        [-0.6422, -0.4384,  0.9792, -0.3993,  1.5593],
        [1.5534,  -0.0542, -4.7256,  0.7244,  2.7350],
        [-0.6864, -2.2435,  0.1585,  0.5304,  3.5411],
        [-0.3190, -1.3160,  0.9689,  1.4887,  0.7079],
        [-0.3200, -0.4654,  0.4880, -0.0559,  0.9622],
        [4.0220,  -3.8886,  1.0547, -0.7427, -0.4464],
        [0.3338,  -0.3306, -0.5961,  1.1220,  0.3529],
        [-0.5572,  0.9190, -0.8745,  2.1899, -0.9497],
    ])
    model.coeff = coeff.unsqueeze(1)  # add extra dim for output vars
    return model


def jang_ex4_data(filename):
    '''
        Read Jang's data for the MG function to be modelled.
    '''
    num_cases = 500
    x = torch.zeros((num_cases, 4))
    y = torch.zeros((num_cases, 1))
    with open(filename, 'r') as fh:
        for i, line in enumerate(fh):
            values = [float(v) for v in line.strip().split()]
            x[i] = torch.tensor(values[0:4])
            y[i] = values[4]
    dl = DataLoader(TensorDataset(x, y), batch_size=1024, shuffle=True)
    return dl


if __name__ == '__main__':
    example = '1'
    show_plots = True
    if len(sys.argv) == 2:  # One arg: example
        example = sys.argv[1]
        show_plots = False
    print('Example {} from Jang\'s paper'.format(example))
    if example == '1':
        model = ex1_model()
        train_data = make_sinc_xy()
        train_anfis(model, train_data, 20, show_plots)
    elif example == '2':
        model = ex2_model()
        train_data = ex2_training_data()
        train_anfis(model, train_data, 200, show_plots)
        test_data = ex2_testing_data()
        test_anfis(model, test_data, show_plots)
    elif example == '3':
        model = ex3_model()
        train_data = ex3_training_data()
        train_anfis(model, train_data, 500, show_plots)
        test_data = ex3_testing_data()
        test_anfis(model, test_data, show_plots)
    elif example == '4':
        model = ex4_model()
        train_data = jang_ex4_data('jang-example4-data.trn')
        train_anfis(model, train_data, 500, show_plots)
        test_data = jang_ex4_data('jang-example4-data.chk')
        test_anfis(model, test_data, show_plots)
    elif example == '4T':
        model = jang_ex4_trained_model()
        test_data = jang_ex4_data('jang-example4-data.trn')
        test_anfis(model, test_data, show_plots)
        test_data = jang_ex4_data('jang-example4-data.chk')
        test_anfis(model, test_data, show_plots)
    else:
        print('ERROR - no such example')
