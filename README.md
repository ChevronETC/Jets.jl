# Jets

Jets is a Julia library for matrix-free linear algebra similar to SPOT
(http://www.cs.ubc.ca/labs/scl/spot/) in Matlab and RVL
(http://www.trip.caam.rice.edu/software/rvl/rvl/doc/html) and Chevron's JLinAlg
in Java.  In addition, Jets is meant as a replacement for Jot
(https://chevron.visualstudio.com/ETC-ESD-Jot).  The purpose is to proved
familiar matrix-vector syntax without forming matrices.  Instead, the action of
the matrix and its adjoint applied to vectors is specified using Julia methods.
In addition, Jets provides a framework for nonlinear functions and their
linearization.  The main construct in this package is a `jet` and is loosely
based on its mathematical namesake
(https://en.wikipedia.org/wiki/Jet_(mathematics)).  In particular, a `jet`
describes a function and its linearization at some point of its domain.

## Companion packages
* DistributedJets - https://chevron.visualstudio.com/ETC-ESD-DistributedJets
* JetPack - https://chevron.visualstudio.com/ETC-ESD-JetPack
* JetPackDSP - http://chevron.visualstudio.com/ETC-ESD-JetPackDSP
* JetPackTransforms - https://chevron.visualstudio.com/ETC-ESD-JetPackTransforms
* JetPackWave - https://chevron.visualstudio.com/ETC-ESD-JetPackWave
* Solvers - https://chevron.visualstudio.com/ETC-ESD-Solvers

## Quick start example
Use SPGL1 in for data reconstruction
```julia
using Pkg
Pkg.add("Jets","JetPackTransforms","Solvers")
using Jets, JetPackTransforms, Solvers
M = JetSpace(Float64,32)
D = JetSpace(Float64,512)
R = JopRestriction(M,D,randperm(D,32))
S = JopFft(R)
t = [0:511;]*6*pi/511
d = R*(sin(t)+cos(t)+2*sin(8t)+2*cos(8t))
m = S'*solve(SolverSpgl1(),R∘S',d)
```

## Vector spaces
The domain and range of the jet are vector spaces.  In Jets, a vector space is represented by one of three concrete types:
```julia
JetSpace <: JetAbstractSpace
JetSSpace <: JetAbstractSpace
JetBSpace <: JetAbstractSpace
```

### JetSpace
`JetSpace` is an n-dimensional vector space with additional meta-data.  The addition meta-data is:
* a **size** `(n₁,n₂,...,nₚ)` where `prod(n₁,n₂,...,nₚ)=n`
* a **type** such as `Float32`, `ComplexF64`, etc.

Examples:
```julia
R₁ = JetSpace(Float32,10) # 10 dimensional space using single precision floats
R₂ = JetSpace(Float64,10,20) # 200 dimensional space with array size 10×20 using double precision floats
R₃ = JetSpace(ComplexF32,10,20,2) # 400 dimensional space with array size 10×20×2 using single precision floats
```

The choice of **shape** and **type** will have various consequences.  For example, using the `rand` convenience function to construct vectors within the space will have the following effects:
```julia
x₁ = rand(R₁) # x₁ will be a 1 dimensional array of length 10 and type Float32
x₂ = rand(R₂) # x₂ will be a 2 dimensional array of size (10,20) and type Float64
x₃ = rand(R₃) # x₃ will be a 3 dimensional array of size (10,20,2) and type ComplexF32
```

### JetSSpace
There are jets that lead to symmetries in its domain/range and that can be sued for increased computational efficiency.  For example, the Fourier transform of a real vector is symmetric.  `JetSSpace` is used to construct an array that include extra information about these symmetries.  In general, jets that have symmetric spaces should provide a method `symspace` for the construction of its symmetric space.

### JetBSpace
Jets provides a block jet with domain and ranges that can be thought of as block vectors.  The domain/range of a block jet is a `JetBSpace`.  For more information, please see the block jet documentation, below.

### Convenience methods for Jet vector spaces
Jets provides the following convenience methods for all types of Jet Vector space `R::JetAbstractSpace`:
```julia
eltype(R) # element type of R
ndims(R) # number of dimensions of arrays in `R`
length(R) # length of arrays in `R` and is equivalent to `prod(size(R))`
size(R) # size of arrays in `R`
reshape(x, R) # reshape `x::AbstractArray` to the size of `R`
ones(R) # array of ones with the element-type and size of `R`
rand(R) # random array with the element-type and size of `R`
zeros(R) # zero array with the element-type and size of `R`
Array(R) # uninitialized array with the element-type and size of `R`
```

## Jets
A jet `J` of type `Jet` is the main construct in this package.  It can be used on its own; but is more often mapped to a linear or nonlinear operators which are described in the following two sections.  We associate the following methods with `jet::Jet`:
```julia
f!(d, jet, m; kwargs...) # function map from domain to range
df!(d, jet, m; kwargs...) # linearized function map from domain to range
df′!(m, jet, d; kwargs...) # linearized adjoint function map from range to domain
domain(jet) # domain of jet
range(jet) # range of jet
eltype(jet) # element-type of the jet
shape(jet) # shape of the domain and range of jet
shape(jet, i) # shape of the range (i=1) or domain (i=2) of jet
size(jet) # size of the domain and range of jet
size(jet,i) # size of the range (i=1) or domain (i=2) of jet
state(jet) # named tuple containing state information of jet
state!(jet, s) # update the state information of jet via the named tuple, s
point(jet) # get the point that the linearization is about
point!(jet, mₒ) # set the point that the linearization is about
close(jet) # closing a jet makes an explicit call to its finalizers
```
Note that the `f!`, `df!` and `df′!` methods are not exported.

## Linear operators

## Differences from Jot
Three new Julia language features that we take advantage of:
1. The Julia abstract array interface.
2. The Julia broadcasting and fused broadcasting feature
3. Named tuples
4. Splatting named tuples into keyword arguments of a function
5. Functions have associated singleton types
