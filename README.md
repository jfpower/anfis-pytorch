# ANFIS in pyTorch #


This is an implementation of the ANFIS system using pyTorch.


### ANFIS

ANFIS is a way of presenting a fuzzy inference system (FIS) as a
series of numeric layers so that it can be trained like a neural net.

The canonical reference is the original paper by
[http://mirlab.org/jang/](Jyh-Shing Roger Jang):

* Jang, J.-S.R. (1993). "ANFIS: adaptive-network-based fuzzy inference
  system". IEEE Transactions on Systems, Man and Cybernetics. 23 (3):
  665–685. doi:10.1109/21.256541

Note that it assumes a Takagi Sugeno Kang (TSK) style of
defuzzification rather than the more usual Mamdani style.


### Implementation

The original C code from Jang that implements the ANFIS system, along
with is test cases, is available from
[https://www.cs.cmu.edu/Groups/AI/areas/fuzzy/systems/anfis/](a repository at CMU).

There's a version of
[https://www.mathworks.com/help/fuzzy/neuro-adaptive-learning-and-anfis.html](ANFIS for Matlab);
the documentation is quite helpful for understanding how ANFIS works,
even if you don't use Matlab.

There's an implementation for R by Cristobal Fresno and Elmer
A. Fernandez of the
[http://www.bdmg.com.ar/?page_id=176](BioScience Data Mining Group)
in Argentina (that URL seems a bit unstable).
Again, their documentation is very helpful, particularly
the "ANFIS vignette" report that comes with the distribution.  It
shows how to run the system using examples from Jang's paper, and gives
some of the results.

I also found a Python implementation of this R code
[https://github.com/twmeggs/anfis](anfis) by twmeggs that was helpful
in understanding the original R code.




### Author ###

* [James Power](http://www.cs.nuim.ie/~jpower/)