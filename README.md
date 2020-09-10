# Jets

Jets is a Julia library for matrix-free linear algebra similar to SPOT
(http://www.cs.ubc.ca/labs/scl/spot/) in Matlab, RVL
(http://www.trip.caam.rice.edu/software/rvl/rvl/doc/html) in C++, and Chevron's JLinAlg
in Java.  In addition, Jets is a successor to Jot
(https://chevron.visualstudio.com/ETC-ESD-Jot).  The purpose of Jets is to provide
familiar matrix-vector syntax without forming matrices.  Instead, the action of
the matrix and its adjoint applied to vectors is specified using Julia methods.
In addition, Jets provides a framework for nonlinear functions and their
linearization.  The main construct in this package is a `jet` and is loosely
based on its mathematical namesake
(https://en.wikipedia.org/wiki/Jet_(mathematics)).  In particular, a `jet`
describes a function and its linearization at some point in its domain.

## Companion packages
* DistributedJets - https://chevron.visualstudio.com/ETC-ESD-DistributedJets.jl
* JetPack - https://chevron.visualstudio.com/ETC-ESD-JetPack.jl
* JetPackDSP - http://chevron.visualstudio.com/ETC-ESD-JetPackDSP.jl
* JetPackTransforms - https://chevron.visualstudio.com/ETC-ESD-JetPackTransforms.jl
* JetPackWave - https://chevron.visualstudio.com/ETC-ESD-JetPackWave.jl
* Solvers - https://chevron.visualstudio.com/ETC-ESD-Solvers.jl

[![](https://img.shields.io/badge/docs-stable-green.svg)](https://chevronetc.github.io/Jets.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://chevronetc.github.io/Jets.jl/dev/)
