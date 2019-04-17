#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: the ANFIS layers
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
    Acknowledgement: twmeggs' implementation of ANFIS in Python was very
    useful in understanding how the ANFIS structures could be interpreted:
        https://github.com/twmeggs/anfis
'''

import itertools
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F


dtype = torch.float


class FuzzifyVariable(torch.nn.Module):
    '''
        Represents a single fuzzy variable, holds a list of its MFs.
        Forward pass will then fuzzify the input (value for each MF).
    '''
    def __init__(self, mfdefs, mfnames=None):
        super(FuzzifyVariable, self).__init__()
        if not mfnames:
            self.mfnames = ['mf{}'.format(i) for i in range(len(mfdefs))]
        else:
            self.mfnames = list(mfnames)
        self.mfdefs = torch.nn.ModuleDict(zip(self.mfnames, mfdefs))
        self.padding = 0

    @property
    def num_mfs(self):
        '''Return the actual number of MFs (ignoring any padding)'''
        return len(self.mfdefs)

    def pad_to(self, new_size):
        '''
            Will pad result of forward-pass (with zeros) so it has new_size,
            i.e. as if it had new_size MFs.
        '''
        self.padding = new_size - len(self.mfdefs)

    def fuzzify(self, x):
        '''
            Yield a list of (mf-name, fuzzy values) for these input values.
        '''
        for mfname, mfdef in self.mfdefs.items():
            yvals = mfdef.forward(x)
            yield(mfname, yvals)

    def forward(self, x):
        '''
            Return a tensor giving the membership value for each MF.
            x.shape: n_cases
            y.shape: n_cases * n_mfs
        '''
        y_pred = torch.cat([mf.forward(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:
            y_pred = torch.cat([y_pred, torch.zeros(x.shape[0], self.padding)], dim=1)
        return y_pred


class FuzzifyLayer(torch.nn.Module):
    '''
        A list of fuzzy variables, representing the inputs to the FIS.
        Forward pass will fuzzify each variable individually.
        We pad the variables so they all seem to have the same number of MFs,
        as this allows us to put all results in the same tensor.
    '''
    def __init__(self, varmfs, varnames=None):
        super(FuzzifyLayer, self).__init__()
        if not varnames:
            self.varnames = ['x{}'.format(i) for i in range(len(varmfs))]
        else:
            self.varnames = list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))

    @property
    def num_in(self):
        '''Return the number of input variables'''
        return len(self.varmfs)

    @property
    def max_mfs(self):
        ''' Return the max number of MFs in any variable'''
        return max([var.num_mfs for var in self.varmfs.values()])

    def show(self):
        '''
            Print the variables, MFS and their parameters (for info only)
        '''
        for varname, members in self.varmfs.items():
            print(varname)
            for mfname, mfdef in members.mfdefs.items():
                print('{}: {}'.format(mfname,
                      [(n, p.item()) for n, p in mfdef.named_parameters()]))

    def forward(self, x):
        ''' Fuzzyify each variable's value using each of its corresponding mfs.
            x.shape = n_cases * n_in
            y.shape = n_cases * n_in * n_mfs
        '''
        assert x.shape[1] == self.num_in, 'wrong no. of input values'
        y_pred = torch.stack([var.forward(x[:, i:i+1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)
        return y_pred


class AntecedentLayer(torch.nn.Module):
    '''
        Form the 'rules' by taking all possible combinations of the MFs
        for each variable. Forward pass then calculates the fire-strengths.
    '''
    def __init__(self, varlist):
        super(AntecedentLayer, self).__init__()
        # Count the (actual) mfs for each variable:
        mf_count = [var.num_mfs for var in varlist]
        # Now make the MF indices for each rule:
        indices = itertools.product(*[range(n) for n in mf_count])
        self.indices = torch.tensor(list(indices))

    def num_rules(self):
        return len(self.indices)

    def show(self, varlist):
        row_ants = []
        mf_count = [len(fv.mfdefs) for fv in varlist.values()]
        for rule_idx in itertools.product(*[range(n) for n in mf_count]):
            thisrule = []
            for (varname, fv), i in zip(varlist.items(), rule_idx):
                thisrule.append('{} is {}'
                                .format(varname, list(fv.mfdefs.keys())[i]))
            row_ants.append(' and '.join(thisrule))
        return row_ants

    def forward(self, x):
        ''' Calculate the fire-strength for (the antecedent of) each rule
            x.shape = n_cases * n_in * n_mfs
            y.shape = n_cases * n_rules
        '''
        # First populate the rule-antecedents with the membership values:
        ants = torch.stack([torch.gather(vals.t(), 0, self.indices)
                            for vals in x])
        # Then take the AND (= product) for each rule-antecedent
        rules = torch.prod(ants, dim=2)
        return rules


class ConsequentLayer(torch.nn.Module):
    '''
        A simple linear layer to represent the TSK consequents
    '''
    def __init__(self, d_in, d_rule, d_out, new_coeff=[]):
        super(ConsequentLayer, self).__init__()
        self.d_in = d_in
        self.d_rule = d_rule
        self.d_out = d_out
        self.coeff = new_coeff

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self._coeff

    @coeff.setter
    def coeff(self, new_coeff):
        '''
            Record the coefficients for all the rules
            coeff: for each rule, for each output variable:
                   a coefficient for each input variable, plus a constant
        '''
        c_shape = torch.Size([self.d_rule, self.d_out, self.d_in+1])
        if len(new_coeff) == 0:  # None given, so set all to zero
            new_coeff = torch.zeros(c_shape, dtype=dtype)
        else:
            assert new_coeff.shape == c_shape, \
                'Coeff shapre should be {}, but is actually {}'\
                .format(c_shape, new_coeff.shape)
        self._coeff = new_coeff

    def fit_coeff(self, x, weights, y_actual):
        '''
            Use LSE to solve for coeff: y_actual = coeff * (weighted)x
                  x.shape: n_cases * n_in
            weights.shape: n_cases * n_rules
            [ coeff.shape: n_rules * n_out * (n_in+1) ]
                  y.shape: n_cases * n_out
        '''
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        # Can't have value 0 for weights, or LSE won't work:
        weights[weights == 0] = 1e-9
        # Shape of weighted_x is n_cases * n_rules * (n_in+1)
        weighted_x = torch.einsum('bp, bq -> bpq', weights, x_plus)
        # Squash x and y down to 2D matrices for gels:
        weighted_x_2d = weighted_x.view(weighted_x.shape[0], -1)
        y_actual_2d = y_actual.view(y_actual.shape[0], -1)
        # Use gels to do LSE, then pick out the solution rows:
        coeff_2d, _ = torch.gels(y_actual_2d, weighted_x_2d)
        coeff_2d = coeff_2d[0:weighted_x_2d.shape[1]]
        # Reshape to 3D tensor: divide by rules, n_in+1, then swap last 2 dims
        self.coeff = coeff_2d.view(weights.shape[1], x.shape[1]+1, -1)\
            .transpose(1, 2)
        # coeff dim is thus: n_rules * n_out * (n_in+1)

    def forward(self, x):
        '''
            Calculate: y = coeff * x + const   [NB: no weights yet]
                  x.shape: n_cases * n_in
              coeff.shape: n_rules * n_out * (n_in+1)
                  y.shape: n_cases * n_out * n_rules
        '''
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        # Need to switch dimansion for the multipy, then switch back:
        y_pred = torch.matmul(self.coeff, x_plus.t())
        return y_pred.transpose(0, 2)  # swaps cases and rules


class WeightedSumLayer(torch.nn.Module):
    '''
        Sum the TSK for each outvar over rules, weighted by fire strengths.
    '''
    def __init__(self):
        super(WeightedSumLayer, self).__init__()

    def forward(self, weights, tsk):
        '''
            weights.shape: n_cases * n_rules
                tsk.shape: n_cases * n_out * n_rules
             y_pred.shape: n_cases * n_out
        '''
        # Add a dimension to weights to ge the bmm to work:
        y_pred = torch.bmm(tsk, weights.unsqueeze(2))
        return y_pred.squeeze(2)


class AnfisNet(torch.nn.Module):
    '''
        This is a container for the 5 layers of the ANFIS net.
        The forward pass maps inputs to outputs based on current settings, 
        and then fit_coeff will adjust the TSK coeff using LSE.
    '''
    def __init__(self, invardefs, outvarnames):
        super(AnfisNet, self).__init__()
        varnames = [v for v, _ in invardefs]
        mfdefs = [FuzzifyVariable(mfs) for _, mfs in invardefs]
        self.num_in = len(invardefs)
        self.num_out = len(outvarnames)
        self.num_rules = np.prod([len(mfs) for _, mfs in invardefs])
        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('fuzzify', FuzzifyLayer(mfdefs, varnames)),
            ('rules', AntecedentLayer(mfdefs)),
            # normalisation layer is just implemented as a function.
            ('consequent', ConsequentLayer(self.num_in, self.num_rules, self.num_out)),
            ('weighted_sum', WeightedSumLayer()),
            ]))

    @property
    def coeff(self):
        return self.layer['consequent'].coeff

    @coeff.setter
    def coeff(self, new_coeff):
        self.layer['consequent'].coeff = new_coeff

    def fit_coeff(self, x, y_actual):
        '''Assumes a forward pass has just been done (so weights available)'''
        self.layer['consequent'].fit_coeff(x, self.weights, y_actual)

    def show_rules(self):
        vardefs = self.layer['fuzzify'].varmfs
        rule_ants = self.layer['rules'].show(vardefs)
        for i, crow in enumerate(self.layer['consequent'].coeff):
            print('Rule {}: IF {}'.format(i, rule_ants[i]))
            print('\tTHEN {}'.format(crow.tolist()))

    def forward(self, x):
        '''
            Forward pass: run x thru the five layers and return the y values.
        '''
        self.fuzzified = self.layer['fuzzify'].forward(x)
        self.raw_weights = self.layer['rules'].forward(self.fuzzified)
        self.weights = F.normalize(self.raw_weights, p=1, dim=1)
        self.rule_tsk = self.layer['consequent'].forward(x)
        y_pred = self.layer['weighted_sum'].forward(self.weights, self.rule_tsk)
        return y_pred

    def forward_debug(self, x, debug_level=99):
        '''
            The foward pass, but print each layer's outputs as you go.
        '''
        if debug_level >= 1:
            print('-'*10, 'Layer 1:', x.shape)
            for i, tc in enumerate(x):
                print('case {}:'.format(i), tc)
            print(x)
        self.fuzzified = self.layer['fuzzify'].forward(x)
        if debug_level >= 2:
            print('-'*10, 'Layer 2:', self.fuzzified.shape)
            for tc_i, tc in enumerate(self.fuzzified):
                for var_i, var in enumerate(tc):
                    print('case {}, x{}: {}'.format(tc_i, var_i, var.tolist()))
        self.raw_weights = self.layer['rules'].forward(self.fuzzified)
        if debug_level >= 3:
            print('-'*10, 'Layer 3:', self.raw_weights.shape)
            print(self.raw_weights)
        self.weights = F.normalize(self.raw_weights, p=1, dim=1)
        if debug_level >= 4:
            print('-'*10, 'Layer 4:', self.weights.shape)
            print(self.weights)
        self.rule_tsk = self.layer['consequent'].forward(x)
        if debug_level >= 5:
            print('-'*10, 'Coefficients', self.layer['consequent'].coeff.shape)
            for r in range(self.num_rules):
                print(self.layer['consequent'].coeff[r].tolist())
            print('-'*10, 'Layer 5:', self.rule_tsk.shape)
            for tc, _ in enumerate(x):
                print(self.rule_tsk[tc].tolist())
        y_pred = self.layer['weighted_sum'].forward(self.weights, self.rule_tsk)
        if debug_level >= 6:
            print('-'*10, 'Layer 6:', y_pred.shape)
            for tc, xvals in enumerate(x):
                print(xvals.tolist(), '->', y_pred[tc].tolist())
        return y_pred
