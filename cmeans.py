#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    Fuzzy c-means (FCM) clustering in torch: standard and SGD version.
    @author: James Power <james.power@mu.ie> May 24 2019
    Looking at cmeans.py from scikit-fuzzy was a big help gor the basic model,
    also "Fuzzy Logic With Engineering Applications" by Timothy J. Ross
        - I had the second edition, so this was in chapter 11, pp 379-389.
'''

import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


dtype = torch.float64

CLUSTER_DATA = 'cluster_data'  # folder where the data sets are stored


class FuzzyCluster(torch.nn.Module):
    '''
        This does fuzzy c-means clustering, maintaining a set of centroids.
        It is packaged as a torch Module, so the forward pass
        re-calculates the partition matrix based on the supplied data.
        Alternate this with recalc_centroids to do the clustering.
        Can optionally register the entroids as parameters and learn them.
    '''

    _EPS = 1e-12  # Value to use instead of 0 (to prevent div-by-zero)

    def __init__(self, n_c, n_in, m=1.7):
        '''
            n_c is the number of clusters
            n_in is the number of input features
            m is the weighting parameter, controls the amount of fuzziness
        '''
        super(FuzzyCluster, self).__init__()
        self.n_c = n_c
        self.n_in = n_in
        self.m = m
        self.centroids = torch.rand((self.n_c, self.n_in), dtype=dtype)
        self.last_u = None  # Record the most recent partition matrix

    def set_centroids(self, new_centroids):
        self.centroids = new_centroids

    def register_centroids(self):
        '''
            Call this to register the centroids as torch parameters,
            so we can (afterwards) use backprop to learn them.
        '''
        init_centroids = self.centroids
        del self.centroids  # Delete the version that's not a Torch parameter
        self.register_parameter('centroids',
                                torch.nn.Parameter(init_centroids))

    @staticmethod
    def _cdist(x1, x2):
        '''
            Pairwise distances between two sets of points
            - this mimics the scipy function spatial.distance.cdist
            Source: https://github.com/pytorch/pytorch/issues/15253
        '''
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        # res = x2^2 -2(x1 @ x2) + x1^2
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1, x2.transpose(-2, -1),
                          alpha=-2).add_(x1_norm)
        res = res.sqrt().clamp_min(FuzzyCluster._EPS)
        return res

    def recalc_centroids(self, x, u):
        '''
            Re-calculate the positions of the centroids for each cluster.
                x.shape = n_cases * n_in
                u.shape = n_cases * self.n_c
            Returns the new centroids (but does not update them):
                v.shape = self.n_c * n_in
        '''
        um = u ** self.m
        # Batch multiply um by v:
        v = torch.einsum('mi,mj->ij', um, x)
        # Divide by u^m, summed by clusters:
        v /= um.sum(dim=0).clamp_min_(FuzzyCluster._EPS).unsqueeze(1)
        return v

    def forward(self, x):
        ''' Calculate and return the partition matrix u,
            which shows (for each x) its membership degree of each cluster
            x.shape = n_cases * n_in
            u.shape = n_cases * self.n_c
        '''
        d = FuzzyCluster._cdist(x, self.centroids)
        u = d ** (- 2. / (self.m - 1))
        self.last_u = u / u.sum(dim=1, keepdim=True)
        return self.last_u


def plot_clusters(x, fc):
    u = fc(x)
    num_clusters = u.shape[1]
    # Hardening: assign each point to cluster with maximum membership:
    crisp_clusters = u.argmax(dim=1)
    # Pick a different color for the points in each cluster:
    all_colors = sns.color_palette("hls", num_clusters)
    pt_colors = [all_colors[i] for i in crisp_clusters]
    # Plot clusters and then centroids:
    plt.figure(1)
    plt.scatter(x[:, 0], x[:, 1], s=1, color=pt_colors)
    centroids = fc.centroids.detach().numpy()
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=175, color='black')


def evaluate_clustering(datset, fc):
    '''
        Compute the fuzzy partition coefficient from u.
        This measures the degree of fuzziness
    '''
    x = dataset.tensors[0]
    u = fc(x)
    trace = torch.einsum('ij,ji->', u, u.t())
    return trace / u.shape[0]


def cmeans_cluster(dataset, num_clusters, max_epochs=250, show_plots=True):
    '''
        Cluster the x data into num_clusters clusters, optionally plot.
        Will execute up to max_epochs, unless centroids stabilise.
        Returns the FuzzyCluster object (contains centroids, partition)
    '''
    MIN_CHANGE = 1e-5  # This is what I mean by "stabilise"
    x = dataset.tensors[0]
    n_in = x.shape[1]
    fc = FuzzyCluster(num_clusters, n_in)
    print('### Training for up to {} epochs, size = {} cases'.
          format(max_epochs, x.shape[0]))
    for t in range(max_epochs):
        u = fc(x)
        new_centroids = fc.recalc_centroids(x, u)
        delta_v = F.l1_loss(new_centroids, fc.centroids)
        fc.set_centroids(new_centroids)
        if max_epochs < 30 or t % 10 == 0:
            print('Epoch {:3d}, change={:.5f}, fuzziness={:.5f}'
                  .format(t, delta_v.item(),
                          evaluate_clustering(dataset, fc)))
        if delta_v < MIN_CHANGE:
            break
    if show_plots:
        plot_clusters(x, fc)
    return fc


def sgd_cluster(dataset, num_clusters, epochs=250, show_plots=True):
    '''
        Cluster the given data into the given number of clusters,
        but use stochastic gradient descent (SGD) and mini-batches.
        Always run for the given number of epochs.
    '''
    BATCH_SIZE = 1024
    data = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    n_in = dataset.tensors[0].shape[1]
    fc = FuzzyCluster(num_clusters, n_in)
    fc.register_centroids()
    optimizer = torch.optim.SGD(fc.parameters(), lr=1e-1, momentum=0.99)
    print('### Training for {} epochs, size = {} cases, batches of {}'.
          format(epochs, dataset.tensors[0].shape[0], BATCH_SIZE))
    for t in range(epochs):
        old_centroids = fc.centroids.clone()
        # Process each mini-batch in turn:
        for x, y_actual in data:
            u = fc(x)
            new_centroids = fc.recalc_centroids(x, u)
            optimizer.zero_grad()
            loss = F.mse_loss(fc.centroids, new_centroids)
            loss.backward()
            optimizer.step()
        # Epoch ending, so print progress for the whole batch:
        with torch.no_grad():
            delta_v = F.l1_loss(old_centroids, fc.centroids)
        if epochs < 30 or t % 10 == 0:
            print('Epoch {:3d}, change={:.5f}, fuzziness={:.5f}'
                  .format(t, delta_v.item(),
                          evaluate_clustering(dataset, fc)))
    # End of training, so graph the results:
    if show_plots:
        plot_clusters(dataset.tensors[0], fc)
    return fc


def read_data(filename, n_in=2):
    '''
        Read the x data points from the given file, one point per line;
        n_in is the number of input features (i.e. co-ords for each point).
        Optionally read the ground-truth categories if there.
        Return a dataset with two tensors: (points, ground-truths)
    '''
    points = []
    truths = []  # i.e. the categories the points actually belong to
    pathname = os.path.join(CLUSTER_DATA, filename)
    with open(pathname, 'r') as fh:
        for line in fh:
            nums = [n for n in line.strip().split()]
            points.append([float(n) for n in nums[:n_in]])
            if len(nums) == n_in+1:
                truths.append(int(nums[n_in]))
    if len(truths) == 0:  # no ground truths supplied
        truths = [-1] * len(points)
    return TensorDataset(torch.tensor(points, dtype=dtype),
                         torch.tensor(truths, dtype=torch.long))


def read_and_cluster(filename, n_c, n_in=2):
    dataset = read_data(filename, n_in)
    fc = cmeans_cluster(dataset, n_c)
    evaluate_clustering(dataset, fc)
    return fc


torch.manual_seed(0)
if __name__ == '__main__':
    example = '7'
    show_plots = True
    if len(sys.argv) == 2:  # One arg: example
        example = sys.argv[1]
        show_plots = False
    if example == '1':
        x = torch.tensor([[1, 3], [1.5, 3.2],  [1.3, 2.8], [3, 1]])
        cmeans_cluster(x, 2)
    elif example == '2':
        read_and_cluster('R15.txt', 15)
    elif example == '3':
        read_and_cluster('Aggregation.txt', 7)
    elif example == '4':
        read_and_cluster('D31.txt', 31)
    elif example == '5':
        read_and_cluster('jain.txt', 5)
    elif example == '6':
        read_and_cluster('a3.txt', 50)
    elif example == '7':
        dataset = read_data('birch3.txt')
        fc = sgd_cluster(dataset, 100, epochs=50)
