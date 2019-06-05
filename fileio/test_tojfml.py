#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: read/write an Anfis system as a JFML object.
    Test cases, exaples.
    @author: James Power <james.power@mu.ie> June 5, 2019
'''

import sys
from collections import OrderedDict

import torch

import anfis
from membership import TrapezoidalMembFunc, TriangularMembFunc

import vignette_examples

from fileio.tojfml import write_xml


def jfml_pendulum_2_domains():
    return {'Angle': (0.0, 255.0), 'ChangeAngle': (0.0, 255.0)}


def jfml_pendulum_2_model():
    '''
        This is a hard-coded version of a JFML example model:
            InvertedPendulum TSK 2, with order-0 and order-1 outputs.
        see: testlib/CreateInvertedPendulumTSKExampleXML2.py in Py4jfml
    '''
    angle = ('Angle', OrderedDict([
            ('very negative', TrapezoidalMembFunc(0.0, 0.0, 48.0, 88.0)),
            ('negative', TriangularMembFunc(48.0, 88.0, 128.0)),
            ('zero', TriangularMembFunc(88.0, 128.0, 168.0)),
            ('positive', TriangularMembFunc(128.0, 168.0, 208.0)),
            ('very positive', TrapezoidalMembFunc(168.0, 208.0, 255.0, 255.0)),
            ]))
    change = ('ChangeAngle', OrderedDict([
            ('very negative', TrapezoidalMembFunc(0.0, 0.0, 48.0, 88.0)),
            ('negative', TriangularMembFunc(48.0, 88.0, 128.0)),
            ('zero', TriangularMembFunc(88.0, 128.0, 168.0)),
            ('positive', TriangularMembFunc(128.0, 168.0, 208.0)),
            ('very positive', TrapezoidalMembFunc(168.0, 208.0, 255.0, 255.0)),
            ]))
    invardefs = [angle, change]
    outvars = ['Force']
    anf = anfis.AnfisNet('JFML inverted pendulum, TSK 2', invardefs, outvars)
    # Enter the coefficients; basically five kinds for the output:
    force = {  # Membership functions for force, as order-1 TSK:
            'vneg': [48.0, 0.01, 0.02],
            'neg':  [88.0, 0.00, 0.00],   # Was order-0, so expanded
            'neu':  [128.0, 0.05, 0.05],
            'pos':  [168.0, 0.00, 0.00],  # Was order-0, so expanded
            'vpos': [208.0, 0.05, 0.03],
            }
    # Now swap each JFML constant term from the front to the back:
    for mfname, mfdef in force.items():
        force[mfname] = mfdef[1:] + mfdef[:1]
    # Finally, enumerate output for all possible input MF conbinations:
    coeffs = torch.tensor([
        force['vneg'],  # RULE1 ang_vneg ca_vneg force_vneg
        force['vneg'],  # RULE1 ang_vneg ca_neg force_vneg
        force['vneg'],  # RULE2 ang_vneg ca_neu force_vneg
        force['neg'],   # RULE3 ang_vneg ca_pos force_neg
        force['neu'],   # RULE4 ang_vneg ca_vpos force_neu

        force['vneg'],  # RULE1 ang__neg ca_vneg force_vneg
        force['vneg'],  # RULE1 ang__neg ca_neg force_vneg
        force['neg'],   # RULE5 ang_neg ca_neu force_neg
        force['neu'],   # RULE6 ang_neg ca_pos force_neu
        force['pos'],   # RULE7 ang_neg ca_vpos force_pos

        force['vneg'],  # RULE8 ang_neu ca_vneg force_vneg
        force['neg'],   # RULE9 ang_neu ca_neg force_neg
        force['neu'],   # RULE10 ang_neu ca_neu force_neu
        force['pos'],   # RULE11 ang_neu ca_pos force_pos
        force['vpos'],  # RULE12 ang_neu ca_vpos force_vpos

        force['neg'],   # RULE13 ang_pos ca_vneg force_neg
        force['neu'],   # RULE14 ang_pos ca_neg force_neu
        force['pos'],   # RULE15 ang_pos ca_neu force_pos
        force['vpos'],  # RULE19 ang_pos_ ca_pos force_vpos
        force['vpos'],  # RULE19 ang_pos_ ca_vpos force_vpos

        force['neu'],   # RULE16 ang_vpos ca_vneg force_neu
        force['pos'],   # RULE17 ang_vpos ca_neg force_pos
        force['vpos'],  # RULE18 ang_vpos ca_neu force_vpos
        force['vpos'],  # RULE19 ang_vpos ca_pos force_vpos
        force['vpos'],  # RULE19 ang_vpos ca_vpos force_vpos
    ])
    # Rules 1 and 19 above involved 'or', so I just expanded them out.
    anf.coeff = coeffs.unsqueeze(1)  # add extra dim for output vars
    return anf


if __name__ == '__main__':
    TEST_FILE = 'test_jfml_out.xml'
    example = '2'
    if len(sys.argv) == 2:  # One arg: example
        example = sys.argv[1]
    if example == '1':
        # Vignette example 5 has two outputs (and uses Gaussian MFs)
        model = vignette_examples.vignette_ex5_trained()
        domains = {'x0': (-10.0, 10.0), 'x1': (-10.0, 10.0)}
        write_xml(model, domains, TEST_FILE)
    elif example == '2':
        model = jfml_pendulum_2_model()
        test_inputs = torch.tensor([
                [0.01, 0.01],
                [10.0, 10.0],
                [125.0, 125.0],
                [250.0, 250.0],
                [254.9, 254.9],
                ])
        outs = model(test_inputs)
        print(outs)
        domains = jfml_pendulum_2_domains()
        write_xml(model, domains, TEST_FILE)
