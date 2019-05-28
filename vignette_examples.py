#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: test cases form the Vignette paper:
        "ANFIS vignette" by Cristobal Fresno and Elmer A. Fernández,
        http://www.bdmg.com.ar/?page_id=176, or CRAN package 'anfis'
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
'''

import sys
import torch

import anfis
from experimental import train_anfis, test_anfis, plot_all_mfs
import jang_examples
from membership import BellMembFunc, GaussMembFunc,\
    make_gauss_mfs, make_bell_mfs, make_tri_mfs, make_trap_mfs

dtype = torch.float


def vignette_ex1():
    '''
        These are the original (untrained) MFS for Vignette example 1.
        Uses 4 Bell membership functions in each input
    '''
    invardefs = [
            ('x0', make_bell_mfs(4, 1, [-10, -3.5, 3.5, 10])),
            ('x1', make_bell_mfs(4, 1, [-10, -3.5, 3.5, 10])),
            ]
    outvars = ['y0']
    anf = anfis.AnfisNet('Vignette Example 1', invardefs, outvars)
    return anf


def vignette_ex2():
    '''
        These are the original (untrained) MFS for Vignette example 2.
        Like example 1, but uses 5 Bell MFs for each input.
    '''
    invardefs = [
            ('x0', make_bell_mfs(4, 1, [-10, -5, 0, 5, 10])),
            ('x1', make_bell_mfs(4, 1, [-10, -5, 5, 0, 10])),
            ]
    outvars = ['y0']
    anf = anfis.AnfisNet('Vignette Example 1', invardefs, outvars)
    return anf


def vignette_ex3():
    '''
        These are the original (untrained) MFS for Vignette example 3.
        Like example 1, but now using 5 Gaussian MFs.
    '''
    invardefs = [
            ('x0', make_gauss_mfs(2, [-10, -5, 0, 5, 10])),
            ('x1', make_gauss_mfs(2, [-10, -5, 0, 5, 10]))
            ]
    outvars = ['y0']
    anf = anfis.AnfisNet('Vignette Example 3', invardefs, outvars)
    return anf


def vignette_ex3a():
    '''
        Not actually from Vignette, but I was just trying triangular Mfs
    '''
    invardefs = [
            ('x0', make_tri_mfs(7.5, [-10, -5, 0, 5, 10])),
            ('x1', make_tri_mfs(7.5, [-10, -5, 0, 5, 10])),
            ]
    outvars = ['y0']
    anf = anfis.AnfisNet('Jang\'s example 1', invardefs, outvars)
    return anf


def vignette_ex3b():
    '''
        Not actually from Vignette, but I was just trying trapezoid Mfs
    '''
    invardefs = [
            ('x0', make_trap_mfs(2, 2, [-10, -5, 0, 5, 10])),
            ('x1', make_trap_mfs(2, 2, [-10, -5, 0, 5, 10])),
            ]
    outvars = ['y0']
    anf = anfis.AnfisNet('Jang\'s example 1', invardefs, outvars)
    return anf


def vignette_ex5():
    '''
        These are the original (untrained) MFS for Vignette example 5
        Same MFs as for example 3, but now there will be two outputs.
        These will be: y0 = sinc(x0, x1) and y1 = 1 - sinc(x0,x1).
    '''
    invardefs = [
            ('x0', make_gauss_mfs(2, [-10, -5, 0, 5, 10])),
            ('x1', make_gauss_mfs(2, [-10, -5, 0, 5, 10]))
            ]
    outvars = ['y0', 'y1']
    anf = anfis.AnfisNet('Vignette Example 5', invardefs, outvars)
    return anf


def vignette_ex1_trained():
    '''
        This is a hard-coded version of Vignette example 1, R version,
        using the mfs/coefficients calculated by R after 57 epochs.
    '''
    invardefs = [
            ('x0', [
                    BellMembFunc(3.939986, 1.628525, -9.979724),
                    BellMembFunc(3.433400, 1.818008, -5.150898),
                    BellMembFunc(3.433400, 1.818008,  5.150898),
                    BellMembFunc(3.939986, 1.628525,  9.979724),
                    ]),
            ('x1', [
                    BellMembFunc(3.939986, 1.628525, -9.979724),
                    BellMembFunc(3.433400, 1.818008, -5.150898),
                    BellMembFunc(3.433400, 1.818008,  5.150898),
                    BellMembFunc(3.939986, 1.628525,  9.979724),
                    ])
            ]
    outvars = ['y0']
    anf = anfis.AnfisNet('Vignette Example 1 (R version)', invardefs, outvars)
    rules = torch.tensor([
        [[-0.03990093, -0.03990093, -0.85724840]],
        [[0.12247975,  -0.02936995,  1.22666375]],
        [[0.12247975,   0.02936995,  1.22666375]],
        [[-0.03990093,  0.03990093, -0.85724840]],
        [[-0.02936995,  0.12247975,  1.22666375]],
        [[0.07627426,   0.07627426,  0.31795799]],
        [[0.07627426,  -0.07627426,  0.31795799]],
        [[-0.02936995, -0.12247975,  1.22666375]],
        [[0.02936995,   0.12247975,  1.22666375]],
        [[-0.07627426,  0.07627426,  0.31795799]],
        [[-0.07627426, -0.07627426,  0.31795799]],
        [[0.02936995,  -0.12247975,  1.22666375]],
        [[0.03990093,  -0.03990093, -0.85724840]],
        [[-0.12247975, -0.02936995,  1.22666375]],
        [[-0.12247975,  0.02936995,  1.22666375]],
        [[0.03990093,   0.03990093, -0.85724840]],
        ], dtype=dtype)
    anf.coeff = rules
    return anf


def vignette_ex5_trained():
    '''
        This is a hard-coded version of Vignette example 3, R version,
        using the mfs/coefficients calculated by R after 10 epochs.
    '''
    invardefs = [
            ('x0', [
                    GaussMembFunc(-9.989877,  2.024529),
                    GaussMembFunc(-4.861332,  2.009401),
                    GaussMembFunc(-5.100757e-12,  1.884703e+00),
                    GaussMembFunc(4.861332, 2.009401),
                    GaussMembFunc(9.989877, 2.024529),
                    ]),
            ('x1', [
                    GaussMembFunc(-9.989877, 2.024529),
                    GaussMembFunc(-4.861332, 2.009401),
                    GaussMembFunc(-7.534084e-13, 1.884703e+00),
                    GaussMembFunc(4.861332, 2.009401),
                    GaussMembFunc(9.989877, 2.024529),
                    ])
            ]
    outvars = ['y0', 'y1']
    anf = anfis.AnfisNet('Vignette Example 5 (R version)', invardefs, outvars)
    y0_coeff = torch.tensor([
      4.614289e-03, 4.614289e-03, 7.887969e-02, -1.349178e-02,
      -9.089431e-03, -1.694363e-01, 7.549623e-02, 5.862259e-14,
      6.962636e-01, -1.349178e-02, 9.089431e-03, -1.694363e-01,
      4.614289e-03, -4.614289e-03, 7.887969e-02, -9.089431e-03,
      -1.349178e-02, -1.694363e-01, 2.645509e-02, 2.645509e-02,
      3.146186e-01, -1.372046e-01, 1.590475e-13, -9.501776e-01,
      2.645509e-02, -2.645509e-02, 3.146186e-01, -9.089431e-03,
      1.349178e-02, -1.694363e-01, 3.138560e-13, 7.549623e-02,
      6.962636e-01, -7.561163e-14, -1.372046e-01, -9.501776e-01,
      -3.100872e-14, -9.339810e-13, 1.363890e+00, -4.795844e-14,
      1.372046e-01, -9.501776e-01, -2.681160e-13, -7.549623e-02,
      6.962636e-01, 9.089431e-03, -1.349178e-02, -1.694363e-01,
      -2.645509e-02, 2.645509e-02, 3.146186e-01, 1.372046e-01,
      1.790106e-13, -9.501776e-01, -2.645509e-02, -2.645509e-02,
      3.146186e-01, 9.089431e-03, 1.349178e-02, -1.694363e-01,
      -4.614289e-03, 4.614289e-03, 7.887969e-02, 1.349178e-02,
      -9.089431e-03, -1.694363e-01, -7.549623e-02, -7.225253e-14,
      6.962636e-01, 1.349178e-02, 9.089431e-03, -1.694363e-01,
      -4.614289e-03, -4.614289e-03, 7.887969e-02,
        ], dtype=dtype).view(25, 3)
    y1_coeff = torch.tensor([
     -1.563721e-02, 1.563721e-02, 7.029522e-01, 6.511928e-03,
     -2.049419e-03, 1.070100e+00, 8.517531e-02, 2.918635e-13,
     -2.147609e-01, 6.511928e-03, 2.049419e-03, 1.070100e+00,
     -1.563721e-02, 1.563721e-02, 7.029522e-01, 2.049419e-03,
     -6.511928e-03, 1.070100e+00, 3.083698e-02, 3.083698e-02,
     -6.477780e-01, 1.310872e-01, 6.044816e-14, 1.928089e+00,
     -3.083698e-02, 3.083698e-02, 6.477780e-01, 2.049419e-03,
     -6.511928e-03, 1.070100e+00, 5.274627e-13, 8.517531e-02,
     -2.147609e-01, 2.688203e-13, 1.310872e-01, 1.928089e+00,
     -3.522058e-15, 9.355811e-13, 3.521679e-01, 1.036118e-13,
     -1.310872e-01, 1.928089e+00, 2.760916e-13, 8.517531e-02,
     -2.147609e-01, 2.049419e-03, 6.511928e-03, 1.070100e+00,
     -3.083698e-02, 3.083698e-02, 6.477780e-01, 1.310872e-01,
     -1.518057e-13, 1.928089e+00, 3.083698e-02, 3.083698e-02,
     -6.477780e-01, 2.049419e-03, 6.511928e-03, 1.070100e+00,
     -1.563721e-02, 1.563721e-02, 7.029522e-01, 6.511928e-03,
     -2.049419e-03, 1.070100e+00, 8.517531e-02, 1.358819e-13,
     -2.147609e-01, 6.511928e-03, 2.049419e-03, 1.070100e+00,
     -1.563721e-02, 1.563721e-02, 7.029522e-01,
        ], dtype=dtype).view(25, 3)
    anf.coeff = torch.stack([y0_coeff, y1_coeff], dim=1)
    return anf


if __name__ == '__main__':
    example = '3a'
    show_plots = True
    if len(sys.argv) == 2:  # One arg: example
        example = sys.argv[1].upper()
        show_plots = False
    print('Example {} from Vignette paper'.format(example))
    if example == '1':
        model = vignette_ex1()
        train_data = jang_examples.make_sinc_xy_large()
        train_anfis(model, train_data, 100, show_plots)
    elif example == '1T':
        model = vignette_ex1_trained()
        test_data = jang_examples.make_sinc_xy()
        test_anfis(model, test_data, show_plots)
    elif example == '2':
        model = vignette_ex2()
        train_data = jang_examples.make_sinc_xy_large(1000)
        train_anfis(model, train_data, 100, show_plots)
    elif example == '3':
        model = vignette_ex3()
        train_data = jang_examples.make_sinc_xy_large()
        train_anfis(model, train_data, 50, show_plots)
    elif example == '3a':
        model = vignette_ex3a()
        train_data = jang_examples.make_sinc_xy_large(1000)
        model.layer.fuzzify.show()
        train_anfis(model, train_data, 250, show_plots)
        model.layer.fuzzify.show()
    elif example == '3b':
        model = vignette_ex3b()
        train_data = jang_examples.make_sinc_xy_large(1000)
        plot_all_mfs(model, train_data.dataset.tensors[0])
        train_anfis(model, train_data, 250, show_plots)
        plot_all_mfs(model, train_data.dataset.tensors[0])
    elif example == '5':
        model = vignette_ex5()
        train_data = jang_examples.make_sinc_xy2()
        train_anfis(model, train_data, 50, show_plots)
    elif example == '5T':
        model = vignette_ex5_trained()
        test_data = jang_examples.make_sinc_xy2()
        test_anfis(model, test_data, show_plots)
    else:
        print('ERROR - no such example')
