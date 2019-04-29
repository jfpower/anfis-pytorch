#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: read/write FIS as text
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
'''

import sys
from functools import reduce

import torch

import anfis
import membership
import jang_examples
import vignette_examples

_ANF_STR = 'ANFIS'
_IN_STR = 'INPUTS'
_MF_STR = 'MEMBERS'
_OUT_STR = 'OUTPUTS'


def _read_comment_line(fh, fstr):
    '''
        Read a line that must start with '# <fstr>',
        return a list of any other (space-separated) elements in the line.
    '''
    line = fh.readline().split()
    assert line[0] == '#', 'Expecting line starting with "#"'
    assert line[1] == fstr, 'Expecting line starting with "# {}"'.format(fstr)
    return line[2:]


def _read_mf(fh):
    '''
        Read a line corresponding to a single MF definition.
        Return an instance of the corresponding MembFunc class.
    '''
    line, comment = fh.readline().split('#')
    line = line.split()
    mfname = line[0]
    mfargs = [float(v) for v in line[1:]]
    # Now use reflection to create an instance of the MembFunc class:
    mfclass = membership.get_class_for[mfname]
    mfinstance = mfclass(*mfargs)
    return mfinstance


def _read_rule(fh):
    '''
        Read a line corresponding to a row of coefficicents (for one rule,var)
        Return a list of float values.
    '''
    line, comment = fh.readline().split('#')
    coeffs = [float(v) for v in line.split()]
    return coeffs


def read(filename):
    '''
        Read a text file with the parameter definitions for an ANFIS model.
        Return an instance of an AnfisNet for it.
    '''
    with open(filename, 'r') as fh:
        mdesc = ' '.join(_read_comment_line(fh, _ANF_STR))
        # Input variables and MFs:
        inames = _read_comment_line(fh, _IN_STR)
        mfnums = [int(v) for v in _read_comment_line(fh, _MF_STR)]
        invardefs = []
        for i, varname in enumerate(inames):
            mfdefs = [_read_mf(fh) for _ in range(mfnums[i])]
            invardefs.append((varname, mfdefs))
        # Output variables and rules:
        outvars = _read_comment_line(fh, _OUT_STR)
        model = anfis.AnfisNet(mdesc, invardefs, outvars)
        coeff_list = []
        rule_count = reduce(lambda x, y: x * y, mfnums, 1)
        for i in range(rule_count):
            rule_i = [_read_rule(fh) for _ in outvars]
            coeff_list.append(rule_i)
        model.coeff = torch.tensor(coeff_list, dtype=torch.float)
    print('Read', filename)
    return model


def show(model, fh=sys.stdout):
    '''
        Write a text representation of a model to the given filehandle.
    '''
    print('#', _ANF_STR, model.description, file=fh)
    # Input variables and MFs:
    invars = model.input_variables()
    print('#', _IN_STR, ' '.join([iname for iname, _ in invars]), file=fh)
    print('#', _MF_STR,
          ' '.join(['{}'.format(fv.num_mfs) for _, fv in invars]),
          file=fh)
    for varname, fv in invars:
        for mfname, mfdef in fv.members():
            print('{}  # {}.{}'.format(mfdef.pretty(), varname, mfname),
                  file=fh)
    # Output variables and rule coefficients:
    onames = model.output_variables()
    print('#', _OUT_STR, ' '.join(onames), file=fh)
    for rnum, rule in enumerate(model.coeff):
        for i, outvar in enumerate(rule):
            ostr = ' '.join(['{}'.format(v) for v in outvar.tolist()])
            print('{}  # {} {}'.format(ostr, onames[i], rnum), file=fh)


def write(model, filename):
    '''
        Write a text representation of a model to the given file.
    '''
    with open(filename, 'w') as fh:
        show(model, fh)
    print('Written', filename)


# Try this out by writing/reading some sample models
if __name__ == '__main__':
    TEST_FILE = 'test-model.txt'
    for model in [jang_examples.jang_ex4_trained_model(),
                  vignette_examples.vignette_ex5_trained()]:
        write(model, TEST_FILE)
        model = read(TEST_FILE)
        show(model)
