#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: read/write an Anfis system as a JFML object.
    Uses Py4JFML: https://github.com/cmencar/py4jfml
    Most particularly, the example:
        Py4jfml/testlib/CreateInvertedPendulumTSKExampleXML1.py
    @author: James Power <james.power@mu.ie> June 5, 2019
'''


from py4jfml.FuzzyInferenceSystem import FuzzyInferenceSystem
from py4jfml.Py4Jfml import Py4jfml
from py4jfml.knowledgebase.KnowledgeBaseType import KnowledgeBaseType
from py4jfml.knowledgebasevariable.FuzzyVariableType import FuzzyVariableType
from py4jfml.knowledgebasevariable.TskVariableType import TskVariableType
from py4jfml.rule.AntecedentType import AntecedentType
from py4jfml.rule.ClauseType import ClauseType
from py4jfml.rule.TskConsequentType import TskConsequentType
from py4jfml.rule.TskFuzzyRuleType import TskFuzzyRuleType
from py4jfml.rulebase.TskRuleBaseType import TskRuleBaseType
from py4jfml.term.FuzzyTermType import FuzzyTermType
from py4jfml.term.TskTermType import TskTermType
from py4jfml.term.TskTerm import TskTerm
from py4jfml.rulebase.FuzzySystemRuleBase import FuzzySystemRuleBase


def _in_mf_def(mfname, mfdef):
    '''
        Return a FuzzyTermType for this MF definition.
    '''
    cname = mfdef.__class__.__name__
    mftype = None
    if cname == 'GaussMembFunc':
        mftype = FuzzyTermType.TYPE_gaussianShape
        param = [mfdef.mu, mfdef.sigma]
    elif cname == 'BellMembFunc':
        param = [mfdef.a, mfdef.b, mfdef.c]
        pass
    elif cname == 'TriangularMembFunc':
        param = [mfdef.a, mfdef.b, mfdef.c]
        mftype = FuzzyTermType.TYPE_triangularShape
    elif cname == 'TrapezoidalMembFunc':
        param = [mfdef.a, mfdef.b, mfdef.c, mfdef.d]
        mftype = FuzzyTermType.TYPE_trapezoidShape
    assert mftype, 'JFML does not implement term type {}'.format(cname)
    param = [p.item() for p in param]  # Values, not torch Parameters
    return FuzzyTermType(name=mfname, type_java=mftype, param=param)


def _out_mf_name(outnum, rnum):
    '''
        Return a made-up name for an output variable's MF (for this rule)
    '''
    return 'LINE_{}_{}'.format(outnum, rnum)


def _out_mf_def(mfname, rule):
    '''
        Return a TskTermType representing this consequent.
        Always an order 1 polynomial, with the coefficients as parameters.
    '''
    coeff = rule.tolist()
    # JFML wants the constant coeff first in the list (not last):
    coeff = coeff[-1:] + coeff[:-1]
    return TskTermType(name=mfname, order=TskTerm._ORDER_1, coeff=coeff)


def _mk_antecedents(local_jfml, rules, invars):
    '''
        Convert the rule antecedents to ClauseType, and combine for each rule.
        Return a list with one AntecedentType per rule.
    '''
    row_ants = []
    for rule_idx in rules.mf_indices:
        thisrule = AntecedentType()
        for (varname, fv), i in zip(invars, rule_idx):
            jfml_var = local_jfml.get_variable(varname)
            mfname = list(fv.mfdefs.keys())[i]
            jfml_term = local_jfml.get_term(varname, mfname)
            ct = ClauseType(jfml_var, jfml_term)
            thisrule.addClause(c=ct)
        row_ants.append(thisrule)
    return row_ants


def _mk_consequents(local_jfml, conseq, onames):
    '''
        Make the rule consequents as (TSK) ThenClauses for each rule.
        Return a list with one ConsequentType per rule.
    '''
    row_cons = []
    for rnum, _ in enumerate(conseq.coeff):
        this_cons = TskConsequentType()
        for outnum, varname in enumerate(onames):
            mfname = _out_mf_name(outnum, rnum)
            this_cons.addTskThenClause(
                    variable=local_jfml.get_variable(varname),
                    term=local_jfml.get_term(varname, mfname))
        row_cons.append(this_cons)
    return row_cons


class _LocalJFML:
    '''
        Keep a local copy of the JFML variables and terms as you build them,
        so that you can reference them later when used in the rules.
        (Basically, a symbol table for variables and their terms).
    '''
    def __init__(self):
        self.variables = {}  # Maps [variable name] to JFML VariableType
        self.terms = {}  # Maps [variable name][mf name] to JFML Term

    def set_in_variable(self, varname, domain):
        jfml_var = FuzzyVariableType(name=varname,
                                     domainLeft=domain[0],
                                     domainRight=domain[1])
        self.variables[varname] = jfml_var
        self.terms[varname] = {}
        return jfml_var

    def set_out_variable(self, varname):
        jfml_var = TskVariableType(name=varname)
        jfml_var.setCombination(value="WA")
        jfml_var.setType(value="output")
        self.variables[varname] = jfml_var
        self.terms[varname] = {}
        return jfml_var

    def get_variable(self, varname):
        return self.variables[varname]

    def set_in_term(self, varname, mfname, mfdef):
        jfml_term = _in_mf_def(mfname, mfdef)
        self.variables[varname].addFuzzyTerm(ft=jfml_term)
        self.terms[varname][mfname] = jfml_term

    def set_out_term(self, varname, mfname, coeffs):
        jfml_term = _out_mf_def(mfname, coeffs)
        self.variables[varname].addTskTerm(jfml_term)
        self.terms[varname][mfname] = jfml_term

    def get_term(self, varname, mfname):
        return self.terms[varname][mfname]


def convert(model, domains):
    '''
        Return a JFML FIS corresponding to this ANFIS model.
        Note you must supply domains (min/max vals) for each input variable.
    '''
    jfml_fis = FuzzyInferenceSystem(model.description)
    jfml_kb = KnowledgeBaseType()
    jfml_fis.setKnowledgeBase(jfml_kb)
    local_jfml = _LocalJFML()
    # Input variables and MFs:
    invars = model.input_variables()
    for varname, fv in invars:
        var = local_jfml.set_in_variable(varname, domains[varname])
        for mfname, mfdef in fv.members():
            local_jfml.set_in_term(varname, mfname, mfdef)
        jfml_kb.addVariable(var)
    # Output variables and rule coefficients:
    onames = model.output_variables()
    for outnum, outvar in enumerate(onames):
        var = local_jfml.set_out_variable(outvar)
        for rnum in range(model.coeff.shape[0]):
            mfname = _out_mf_name(outnum, rnum)
            local_jfml.set_out_term(outvar, mfname, model.coeff[rnum][outnum])
        jfml_kb.addVariable(var)
    # Rules (all in one rule block)
    jfml_rb = TskRuleBaseType(name="Rulebase",
                              tskRuleBaseType=FuzzySystemRuleBase.TYPE_TSK)
    jfml_rb.setActivationMethod(value="PROD")
    rule_ants = _mk_antecedents(local_jfml, model.layer['rules'], invars)
    rule_cons = _mk_consequents(local_jfml, model.layer['consequent'], onames)
    for rnum, rule in enumerate(model.coeff):
        jfml_rule = TskFuzzyRuleType(name="rule{}".format(rnum),
                                     connector="and",
                                     connectorMethod="MIN", weight=1.0)
        jfml_rule.setAntecedent(value=rule_ants[rnum])
        jfml_rule.setTskConsequent(value=rule_cons[rnum])
        jfml_rb.addTskRule(rule=jfml_rule)
    jfml_fis.addRuleBase(jfml_rb)
    return jfml_fis


def write_xml(model, domains, filename):
    '''
        Write an XML representation of an ANFIS model to the given file.
    '''
    jfml_fis = convert(model, domains)
    Py4jfml.writeFSTtoXML(jfml_fis, filename)
    print('Written', filename)
