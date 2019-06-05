#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: read/write an Anfis system as a FCL file.
    I'm not fully sure how FCL deals with TSK antecedents, if at all.
    The proper FCL specification is IEC document 61131-7,
    but I'm working from Committee Draft CD 1.0 (Rel. 19 Jan 97).
        http://www.fuzzytech.com/binaries/ieccd1.pdf
    @author: James Power <james.power@mu.ie> June 4, 2019
'''

import sys

_COMMENT = '//'
_INDENT = '  '
_BLANKLINE = '\n'
_SEMIC = ';'  # Make this empty if you don't want a semic at the end of lines

_CONJ_STR = ' AND '    # Join antecedent and consequent clauses
_VARTYPE = 'REAL'      # Type of the input/output variables
_TSK_COEFF = 'Linear'  # How I list TSK-style coefficients in the MFs


def _in_mf_def(mfdef):
    '''
        Define bespoke translations for the membership functions.
        Return a string with the MF name and parameters.
        Based on the jfuzzylogic names,
          http://jfuzzylogic.sourceforge.net/html/manual.html#membership
    '''
    cname = mfdef.__class__.__name__
    if cname == 'GaussMembFunc':
        return 'gauss {} {}'.format(mfdef.mu, mfdef.sigma)
    elif cname == 'BellMembFunc':
        return'gbell {} {} {}'.format(mfdef.a, mfdef.b, mfdef.c)
    elif cname == 'TriangularMembFunc':
        return'trian {} {} {}'.format(mfdef.a, mfdef.b, mfdef.c)
    elif cname == 'TrapezoidalMembFunc':
        return'trape {} {} {} {}'.format(mfdef.a, mfdef.b, mfdef.c, mfdef.d)
    else:
        return mfdef.pretty()


def _out_mf_def(rule):
    '''
        This is how we represent an output (TSK) membership function.
    '''
    params = ' '.join(['{}'.format(coeff) for coeff in rule.tolist()])
    return '{} {}'.format(_TSK_COEFF, params)


def _out_mf_name(outnum, rnum):
    '''
        Return a made-up name for an output variable's MF (for this rule)
    '''
    return 'LINE_{}_{}'.format(outnum, rnum)


def _show_antecedents(rules, invars):
    '''
        Depict the rule antecedents as "v1 is mf1 AND v2 is mf2 AND ..."
        Return a list with one entry per rule.
    '''
    row_ants = []
    for rule_idx in rules.mf_indices:
        thisrule = []
        for (varname, fv), i in zip(invars, rule_idx):
            thisrule.append('{} is {}'
                            .format(varname, list(fv.mfdefs.keys())[i]))
        row_ants.append(_CONJ_STR.join(thisrule))
    return row_ants


def show(model, fh=sys.stdout):
    '''
        Write a text representation of a model to stdout or given filehandle.
    '''
    print('FUNCTION_BLOCK', _COMMENT, model.description, file=fh)
    print(_BLANKLINE, file=fh)
    # Input variables and MFs:
    invars = model.input_variables()
    print('VAR_INPUT', file=fh)
    for varname, _ in invars:
        print(_INDENT, '{} : {}'.format(varname, _VARTYPE), _SEMIC, file=fh)
    print('END_VAR', _BLANKLINE, file=fh)
    for varname, fv in invars:
        print('FUZZIFY', varname, file=fh)
        for mfname, mfdef in fv.members():
            print(_INDENT, 'TERM {} := {}'.format(mfname, _in_mf_def(mfdef)),
                  _SEMIC, file=fh)
        print('END_FUZZIFY', _BLANKLINE, file=fh)
    print(_BLANKLINE, file=fh)
    # Output variables and rule coefficients:
    onames = model.output_variables()
    print('VAR_OUTPUT', file=fh)
    for varname in onames:
        print(_INDENT, '{} : {}'.format(varname, _VARTYPE), _SEMIC, file=fh)
    print('END_VAR', _BLANKLINE, file=fh)
    for outnum, outvar in enumerate(onames):
        print('DEFUZZIFY', outvar, file=fh)
        for rnum in range(model.coeff.shape[0]):
            mfname = _out_mf_name(outnum, rnum)
            mfdef = _out_mf_def(model.coeff[rnum][outnum])
            print(_INDENT, 'TERM LINE_{}_{} := {}'.format(
                  outnum, rnum, mfdef), _SEMIC, file=fh)
        print('END_DEFUZZIFY', _BLANKLINE, file=fh)
    print(_BLANKLINE, file=fh)
    # Rules (all in one rule block)
    rule_ants = _show_antecedents(model.layer['rules'], invars)
    print('RULEBLOCK', file=fh)
    for rnum, rule in enumerate(model.coeff):
        print(_INDENT, 'RULE {}: IF {}'.format(rnum, rule_ants[rnum]), file=fh)
        conseq = []
        for outnum, outvar in enumerate(rule):
            mfname = _out_mf_name(outnum, rnum)
            conseq.append('{} is {}'.format(onames[outnum], mfname))
        print(_INDENT*2, 'THEN {}'.format(_CONJ_STR.join(conseq)),
              _SEMIC, file=fh)
    print('END_RULEBLOCK', _BLANKLINE, file=fh)
    print(_BLANKLINE, file=fh)
    print('END_FUNCTION_BLOCK', file=fh)


def write(model, filename):
    '''
        Write a text representation of a model to the given file.
    '''
    with open(filename, 'w') as fh:
        show(model, fh)
    print('Written', filename)
