#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: some simple test cases
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
"""

import torch

import anfis

from membership import make_gauss_mfs, BellMembFunc


def simple_tests():
    test_mfs = [
        make_gauss_mfs(2, [-10, -5, 0, 5, 10]),
        make_gauss_mfs(5, [-7.5, 0, 7.5]),
    ]

    test_x = torch.rand((121, 1))
    # print([mf.forward(test_x) for mf in test_mfs[0]])

    test_vars = [anfis.FuzzifyVariable(mfs) for mfs in test_mfs]
    test_vars[0].forward(test_x).shape

    test_xs = torch.rand((121, 2))
    fl = anfis.FuzzifyLayer(test_vars, ['food', 'service'])
    in_l2 = fl.forward(test_xs)

    al = anfis.AntecedentLayer(fl.varmfs.values())
    in_l3 = al.forward(in_l2)

    return in_l3


def tipper_ex(batch_size=121):
    invardefs = [
            ('happiness', make_gauss_mfs(5, [-5, 5])),
            ('food', make_gauss_mfs(2, [-10, -5, 0, 5, 10])),
            ('service', make_gauss_mfs(5, [-7.5, 0, 7.5])),
            ]
    outvars = ['tip1', 'tip2']
    anf = anfis.AnfisNet('Tipper', invardefs, outvars)
    test_xs = torch.rand((batch_size, len(invardefs)))
    y_pred = anf.forward(test_xs)
    return y_pred


def vignette_ex1_py():
    '''
        This is a hard-coded version of Vignette example 1, Python version
        using the mfs/coefficients calculated by my anfis after 27 epochs.
        These are almost certainly wrong.
    '''
    invardefs = [
            ('x0', [
                    BellMembFunc(a=4.096336, b=1.424759, c=-9.859497),
                    BellMembFunc(a=3.599561, b=1.027690, c=-4.177343),
                    BellMembFunc(a=3.599561, b=1.027690, c=4.177343),
                    BellMembFunc(a=4.096336, b=1.424759, c=9.859497),
                    ]),
            ('x1', [
                    BellMembFunc(a=4.096336, b=1.424759, c=-9.859497),
                    BellMembFunc(a=3.599561, b=1.027690, c=-4.177343),
                    BellMembFunc(a=3.599561, b=1.027690, c=4.177343),
                    BellMembFunc(a=4.096336, b=1.424759, c=9.859497),
                    ])
            ]
    outvars = ['y0']
    anf = anfis.AnfisNet('', invardefs, outvars)
    rules = torch.tensor([
            [[-0.05, -0.05, -1.40]],
            [[-0.05, -1.40, +0.11]],
            [[-1.40, +0.11, -0.10]],
            [[+0.11, -0.10, +1.29]],
            [[-0.10, +1.29, +0.11]],
            [[+1.29, +0.11, +0.10]],
            [[+0.11, +0.10, +1.29]],
            [[+0.10, +1.29, -0.05]],
            [[+1.29, -0.05, +0.05]],
            [[-0.05, +0.05, -1.40]],
            [[+0.05, -1.40, -0.10]],
            [[-1.40, -0.10, +0.11]],
            [[-0.10, +0.11, +1.29]],
            [[+0.11, +1.29, +0.17]],
            [[+1.29, +0.17, +0.17]],
            [[+0.17, +0.17, +0.38]],
            ], dtype=torch.float)
    anf.set_rules(rules)
    return anf
