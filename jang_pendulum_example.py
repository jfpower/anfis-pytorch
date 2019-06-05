#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: Control examples from Jang's book, chapter 17.
        Section 17.6.2: Recurrent learning, inverted pendulum case study.
    or "Self Learning of Fuzzy Controllers Based on Temporal Back Propagation"
        IEEE Trans on Neural Networks 3(5), Sept 1992.
    @author: James Power <james.power@mu.ie> May 8 2019
'''

# The PendulumSystem module is based very roughly on:
#   #pytorch-control-flow-weight-sharing from
#   https://pytorch.org/tutorials/beginner/pytorch_with_examples.html


import numpy as np
import matplotlib.pyplot as plt

import torch

import anfis
from membership import make_bell_mfs
import fileio.astext

dtype = torch.float
np.random.seed(0)


class Pendulum():
    '''
        Represents the physical model of the pendulum.
        The internal state is a pair (theta, dtheta) - or a tensor of these.
        The mass of the cart and mass/length of the pole are hardwired.
    '''
    def __init__(self, theta=0, dtheta=0):
        '''
            Set up the pendulum; the initial state is (theta, dtheta)
            Assume theta/dtheta are measured *in degrees*.
        '''
        self._state = torch.tensor((theta, dtheta), dtype=dtype).reshape(1, 2)
        self.m_c = 1.0  # mass of cart in kg
        self.m = 0.1  # mass of pole in kg
        self.len = 0.5  # half-length of pole, in m

    _g = 9.81  # acceleration due to gravity in m/s

    @property
    def theta(self):
        return self._state[:, 0]

    @property
    def dtheta(self):
        return self._state[:, 1]

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state

    def _theta_dot_dot_radians(self, rtheta, rdtheta, force):
        '''
            The physics bit: use the diff equations to calculate ddtheta
            N.B. all angles here (theta, dtheta, ddtheta) are in radians.
        '''
        tot_mass = self.m_c + self.m
        numer = Pendulum._g * torch.sin(rtheta) + torch.cos(rtheta) * (
            (-force - (self.m * self.len * rdtheta**2 * torch.sin(rtheta)))
            / tot_mass)
        denom = self.len * ((4./3.) -
                            ((self.m * torch.cos(rtheta)**2) / tot_mass))
        return (numer / denom)

    def theta_dot_dot(self, force):
        '''
            Calculate and return ddtheta (assume we're working in degrees).
        '''
        rtheta = self.theta * (np.pi / 180.)
        rdtheta = self.dtheta * (np.pi / 180.)
        rddtheta = self._theta_dot_dot_radians(rtheta, rdtheta, force)
        return rddtheta * (180. / np.pi)

    def take_step(self, force, h=10e-3):
        '''
            Update theta/dtheta to new values based on given force.
            h is the step size in seconds (so, default is 10ms).
            For convenience, return the current state.
        '''
        ddtheta = self.theta_dot_dot(force.squeeze(1))  # uses current state
        delta = torch.stack((self.dtheta, ddtheta), dim=1)
        self._state = self._state + (h * delta)
        return self.state


def initial_anfis():
    '''
        Build and return a (non-trained) anfis model: (theta, dtheta) -> force
        Assume range for theta is (-20, 20) and dtheta is (-50, 50)
        Use 2 Bell MFs for each input, and non-hybrid learning.
    '''
    invardefs = [
            ('theta', make_bell_mfs(20, 2, [-20, 20])),
            ('dtheta', make_bell_mfs(50, 2, [-50, 50])),
            ]
    outvars = ['force']
    anf = anfis.AnfisNet('Pendulum Controller',
                         invardefs, outvars, hybrid=False)
    return anf


def jang_traned_anfis():
    '''
        This is the trained ANFIS model from Jang's book (pg 474)
    '''
    invardefs = [
            ('theta', make_bell_mfs(-1.59, 2.34, [-19.49, 19.49])),
            ('dtheta', make_bell_mfs(85.51, 1.94, [-23.21, 23.21])),
            ]
    outvars = ['force']
    coeffs = torch.tensor([
            [0.0502, 0.1646, -10.09],
            [0.0083, 0.0119, -1.09],
            [0.0083, 0.0119, 1.09],
            [0.0502, 0.1646, 10.09],
            ], dtype=dtype).unsqueeze(1)
    anf = anfis.AnfisNet('Pendulum Controller',
                         invardefs, outvars, hybrid=False)
    anf.coeff = coeffs
    return anf


class PendulumSystem(torch.nn.Module):
    '''
        The pendulum system consists of an ANFIS controller and a pendulum.
        We make one copy of the ANFIS controller for each time interval.
        But: only one ANFIS object, so only one set of parameters to train.
    '''
    def __init__(self, theta=0, dtheta=0):
        super(PendulumSystem, self).__init__()
        self.anfis = initial_anfis()
        self.pendulum = Pendulum(theta, dtheta)
        self.interval = 100  # Actually the number of time intervals

    def forward(self, x):
        '''
            Run the anfis/pendulum pairing self.interval times.
            Return a tensor of the trajectory: (theta, dtheta, force) values.
                x.shape: n_cases * 2   (== pendulum.state.shape)
                force.shape: n_cases * 1
                this_pass.shape: n_cases * 3  (= theta, dtheta, force)
                trajectory.shape: n_cases * 3 * self.interval
        '''
        # Create an empty trajectory first, and then fill in the values:
        trajectory = torch.empty((x.shape[0], 3, self.interval))
        self.pendulum.state = x
        for i in range(self.interval):
            # First run the anfis to get the force, then apply to pendulum:
            force = self.anfis(self.pendulum.state)
            self.pendulum.take_step(force)
            # Make trajectory for this pass, and store it in the result:
            this_pass = torch.cat((self.pendulum.state, force), dim=1)
            trajectory[:, :, i] = this_pass
        return trajectory


def loss_from(trajectory, desired_trajectory, lam=10):
    '''
        This is a more generalised loss function for the pendulum system.
        It's basically a combination of the (sum-squared) theta and force.
        We minimise force, so the force target is always 0 (no subtraction).
        The parameter lam(bda) is the weigting given to minimising force.
            trajectory.shape: n_cases * 3 * self.interval
    '''
    traj_err = torch.sum((trajectory[:, 0]-desired_trajectory)**2, dim=1)
    force_err = torch.sum(trajectory[:, 2]**2, dim=1)
    sum_sq_err = traj_err + (lam * force_err)
    # I average these over all the input cases:
    return torch.mean(sum_sq_err)


def loss_from_upright(trajectory, lam=10):
    '''
        This is the default loss function for the pendulum system.
        Target is zero angle and zero force.
            trajectory.shape: n_cases * 3 * self.interval
    '''
    # Desired trajectory is just zero always:
    desired_trajectory = torch.zeros(trajectory.shape[2])
    return loss_from(trajectory, desired_trajectory, lam)


def plot_errors(errors):
    '''
        Plot the given list of error rates against no. of epochs
    '''
    plt.plot(range(len(errors)), errors, '-ro', label='errors')
    plt.ylabel('Training Error')
    plt.xlabel('Epoch')
    plt.show()


def plot_thetas(x_data, y_pred):
    '''
        Plot the predicted values for theta (should go towards zero)
    '''
    # Plot the zero line:
    plt.hlines(y=0, xmin=0, xmax=y_pred.shape[2], linestyle=':', color='grey')
    for i in range(y_pred.shape[0]):
        init_theta = x_data[i][0]
        init_dtheta = x_data[i][1]
        legend = 'TC{}: ({}, {})'.format(i, init_theta, init_dtheta)
        thetas = y_pred[i, 0, :].tolist()
        plt.plot(range(len(thetas)), thetas, 'b', label=legend)
        plt.legend(loc='upper right')
        plt.xlabel('Time in 10ms intervals')
        plt.ylabel('Theta in degrees')
    plt.show()


def train_pendulum(model, x_data, optimizer,
                   epochs=500, show_plots=False,
                   loss_lambda=10):
    '''
        Train the given model using the given (x,y) data.
    '''
    errors = []  # Keep a list of these for plotting afterwards
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(x_data.shape)
    print('### Training for {} epochs, training size = {} cases'.
          format(epochs, x_data.shape[0]))
    for t in range(epochs):
        y_pred = model(x_data)
        # Compute and print loss
        loss = loss_from_upright(y_pred, loss_lambda)
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        errors.append(loss.item())
        # Print some progress information as the net is trained:
        if epochs < 30 or t % 10 == 0:
            print('epoch {:4d}: loss={:.5f}'.format(t, loss.item()))
    # End of training, so graph the results:
    if show_plots:
        plot_errors(errors)
        y_pred = model(x_data)
        plot_thetas(x_data, y_pred)


if __name__ == '__main__':
    model = PendulumSystem()
    want_training = True
    if want_training:
        print('### TRAINING ###')
        training_data = torch.tensor([[10, 10], [-10, 0]], dtype=dtype)
        optimizer = torch.optim.Rprop(model.parameters(), lr=1e-2)
        train_pendulum(model, training_data, optimizer, 3, True)
    else:  # Use the following if you want to use Jang's trained model:
        model.anfis = jang_traned_anfis()

    print('### TESTING ###')
    test_data = torch.tensor([[10, 20], [15, 30], [20, 40]], dtype=dtype)
    model.interval = 200
    y_pred = model(test_data)
    plot_thetas(test_data, y_pred)

    print('### TRAINED MODEL ###')
    fileio.astext.show(model.anfis)
