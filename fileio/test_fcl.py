#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:50:44 2019

@author: jpower
"""
import jang_examples
import vignette_examples
from fileio.fcl import show

# Try out text IO by writing/reading some sample models
testcases = [jang_examples.jang_ex4_trained_model(),
             vignette_examples.vignette_ex5_trained()]
if __name__ == '__main__':
    TEST_FILE = 'test-model.txt'
    for model in testcases:
        show(model)
