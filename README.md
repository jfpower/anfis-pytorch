# ANFIS in pyTorch #


This is an implementation of the ANFIS system using pyTorch.


### ANFIS

ANFIS is a way of presenting a fuzzy inference system (FIS) as a
series of numeric layers so that it can be trained like a neural net.

The canonical reference is the original paper by
[Jyh-Shing Roger Jang](http://mirlab.org/jang/):

* Jang, J.-S.R. (1993). "ANFIS: adaptive-network-based fuzzy inference
  system". IEEE Transactions on Systems, Man and Cybernetics. 23 (3):
  665–685. doi:10.1109/21.256541

Note that it assumes a Takagi Sugeno Kang (TSK) style of
defuzzification rather than the more usual Mamdani style.


### Background: other implementations

The original C code from Jang that implements the ANFIS system, along
with is test cases, is available from
[a repository at CMU](https://www.cs.cmu.edu/Groups/AI/areas/fuzzy/systems/anfis/).

There's a version of
[ANFIS for Matlab](https://www.mathworks.com/help/fuzzy/neuro-adaptive-learning-and-anfis.html);
the documentation is quite helpful for understanding how ANFIS works,
even if you don't use Matlab.

There's an implementation for the R language by Cristobal Fresno and Elmer
A. Fernandez of the
[BioScience Data Mining Group](http://www.bdmg.com.ar/?page_id=176)
in Argentina (that URL seems a bit unstable, so I'll put a
[local copy](./Anfis-vignette.pdf)
here).
Again, their documentation is very helpful, particularly
the "ANFIS vignette" report that comes with the distribution.  It
shows how to run the system using examples from Jang's paper, and gives
some of the results.

I also found a Python implementation of this R code
[anfis](https://github.com/twmeggs/anfis) by twmeggs that was helpful
in understanding the original R code.


### Navigation

* [test-anfis.py](./test-anfis.py) If you can run this, then everything should be
  working.  This file contains some test cases from Jang's paper.
  
* [anfis.py](./anfis.py) This is where the layers of the ANFIS system are defined as
  Torch modules.

* [membership.py](./membership.py) At the moment I only have Bell and Gaussian membership
  functions, but any others will go in here too.

* [experimental.py](./experimental.py) Just some simple routines to load data and plot graphs etc.

* [simple-tests.py](./simple-tests.py) Even simpler test cases, you
  can safely ignore these.


### Installation

You need to install Python and PyTorch, nothing special.

I'm using
(Python 3.6.5](https://www.python.org/downloads/),
the (Anaconda 4.6.11)[https://www.anaconda.com/distribution/] stack
and [Torch](https://pytorch.org) version 1.0.1.


### Author ###

* [James Power](http://www.cs.nuim.ie/~jpower/)