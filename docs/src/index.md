# Jets

Jets is a Julia library for matrix-free linear algebra and nonlinear optimization.

Other Julia packages that provide similar functionality include:
- LinearMaps - <https://github.com/Jutho/LinearMaps.jl>
- FunctionalOperators - <https://github.com/hakkelt/FunctionOperators.jl>
- AbstractOperators - <https://github.com/kul-forbes/AbstractOperators.jl>
- JOLI - <https://github.com/slimgroup/JOLI.jl>
- BlockArrays - <https://github.com/JuliaArrays/BlockArrays.jl> 

The purpose of Jets is to provide familiar matrix-vector syntax without forming matrices. Instead, the action of the matrix and its adjoint applied to vectors is specified using Julia methods. In addition, Jets provides a framework for nonlinear functions and their linearization. The main construct in this package is a `jet` and is loosely based on its mathematical namesake (<https://en.wikipedia.org/wiki/Jet_(mathematics)>). In particular,  a `jet` describes a function and its linearization at some point in its domain.

## Companion packages in the COFII framework
- DistributedJets - <https://github.com/ChevronETC/DistributedJets.jl>
- JetPack - <https://github.com/ChevronETC/JetPack.jl>
- JetPackDSP - <https://github.com/ChevronETC/JetPackDSP.jl>
- JetPackWaveFD - <https://github.com/ChevronETC/JetPackWaveFD.jl>
- JetPackTransforms - <https://github.com/ChevronETC/JetPackTransforms.jl>

## Vector spaces
The domain and range of a jet are vector spaces. In `Jets`, a vector space is represented by one of three concrete types:
```julia
JetSpace <: JetAbstractSpace
JetSSpace <: JetAbstractSpace
JetBSpace <: JetAbstractSpace
```

### JetSpace
`JetSpace` is an n-dimensional vector space with additional meta-data. The addition meta-data is:
* a **size** `(n₁,n₂,...,nₚ)` where `prod(n₁,n₂,...,nₚ)=n`
* a **type** such as `Float32`, `Complex{Float64}`, etc.

**Examples**
```julia
R₁ = JetSpace(Float32, 10)           # 10 dimensional space using single precision floats
R₂ = JetSpace(Float64, 10, 20)       # 200 dimensional space with array size 10×20 using double precision floats
R₃ = JetSpace(ComplexF32, 10, 20, 2) # 400 dimensional space with array size 10×20×2 using single precision floats
```

The choice of **shape** and **type** will have various consequences. For example, using the `rand` convenience function to construct vectors within the space will have the following effects:
```julia
x₁ = rand(R₁) # x₁ will be a 1 dimensional array of length 10 and type Float32
x₂ = rand(R₂) # x₂ will be a 2 dimensional array of size (10,20) and type Float64
x₃ = rand(R₃) # x₃ will be a 3 dimensional array of size (10,20,2) and type ComplexF32
```

### JetSSpace
There are jets that lead to symmetries in their domain/range, and those symmetries can be used for increased computational efficiency. For example, the Fourier transform of a real vector has Hermitian symmetry for negative frequencies. `JetSSpace` is used to construct an array that includes extra information about these symmetries. In general, jets that have symmetric spaces should provide a method `symspace` for the construction of its symmetric space. 

**Example**
```julia
using Pkg
Pkg.add("Jets","JetPackFft")
A = JopFft(JetSpace(Float64,128))
R = range(A) # the range of A is a symmetric space R<:JetSSpace
```

### JetBSpace
`Jets` provides a block jet that is analogous to a block matrix. The domain and range associated with a block jet is a `JetBSpace`, and a `JetBSpace` adds book-keeping information to describe this blocked structure. For more information, please see the block jet documentation, below.

### Convenience methods for Jet vector spaces
Jets provides the following convenience methods for all concrete Jet Vector spaces `R::JetAbstractSpace`:
```julia
eltype(R)     # element type of R
ndims(R)      # number of dimensions of arrays in `R`
length(R)     # length of arrays in `R` and is equivalent to `prod(size(R))`
size(R)       # size of arrays in `R`
reshape(x, R) # reshape `x::AbstractArray` to the size of `R`
ones(R)       # array of ones with the element-type and size of `R`
rand(R)       # random array with the element-type and size of `R`
zeros(R)      # zero array with the element-type and size of `R`
Array(R)      # uninitialized array with the element-type and size of `R`
vec(R)        # return a similar space, but backed by a one dimensional array
```

## Jets
A jet `jet::Jet` is the main construct in this package. It can be used on its own; but is more often wrapped in a linear or nonlinear operator which will be discussed shortly. We associate the following methods with `jet::Jet`:
```julia
f!(d, jet, m; kwargs...)   # function map from domain to range
df!(d, jet, m; kwargs...)  # linearized function map from domain to range
df′!(m, jet, d; kwargs...) # linearized adjoint function map from range to domain
domain(jet)                # domain of jet
range(jet)                 # range of jet
eltype(jet)                # element-type of the jet
shape(jet)                 # shape of the domain and range of jet
shape(jet, i)              # shape of the range (i=1) or domain (i=2) of jet
size(jet)                  # size of the domain and range of jet
size(jet,i)                # size of the range (i=1) or domain (i=2) of jet
state(jet)                 # named tuple containing state information of jet
state!(jet, s)             # update the state information of jet via the named tuple, s
point(jet)                 # get the point that the linearization is about
point!(jet, mₒ)            # set the point that the linearization is about
close(jet)                 # closing a jet makes an explicit call to its finalizers
```
Note that the `f!`, `df!` and `df′!` methods are not exported.

**Example, creating a jet for the function `f(x)=x^a`**
```julia
using Pkg
Pkg.add("Jets")
foo!(d, m; a, kwargs...) = d .= x.^a
dfoo!(δd, δm; mₒ, a, kwargs...) = δd .= a * mₒ.^(a-1) .* δm
jet = Jet(dom = JetSpace(Float64,128), rng = 
    JetSpace(Float64,128), f! = foo!, df! = dfoo!, s = (a=1.0,))
```

In the above construction, we define the domain (`dom`), range (`rng`), and a function (`f!`) with its linearization (`df!`). In addition, the jet contains *state*. In this case the state is the value of the exponent `a`. The state is passed to the jet using the named tuple `s = (a=1.0,)`. Notice that construction of the jet uses Julia's named arguments. 

Finally, we note that for this specific example, the construction does not specify the adjoint of the lineariziation. This is because for this specific case the linearization is self-adjoint. An equivalent construction that explicitly includes the adjoint is:
```julia
jet = Jet(dom = JetSpace(Float64,128), rng = 
    JetSpace(Float64,128), f! = foo!, df! = dfoo!, df′! = dfoo!, s=(a=1.0,))
```

## Linear and nonlinear operators
A jet can be wrapped into nonlinear (`JopNl`) and linear (`JopLn`) operators. When we wrap a nonlinear operator around a jet, we must also specify the point at which we linearize. Continuing from the `jet` defined in the previous section, we first show a linear operator and then a nonlinear operator.

**Example: linear operator**
```julia
A = JopLn(jet, rand(domain(A))  # A is a linear operator linearized about a random point in domain(A)
m = rand(domain(A))             # m is a vector in domain(A)
d = A*m                         # d is a vector in range(A), computed via the dfoo! method
mul!(d, A, m)                   # equivalent in-place version of the previous line
a = A'*d                        # a is a vector in domain(A), computed via dfoo! (A is self-adjoint)
mul!(a, A', d)                  # equivalent in-place version of the previous line
```

**Example: nonlinear operator**
```julia
F = JopNl(jet)                  # F is a nonlinear operator
m = rand(domain(A))             # m is a vector in domain(A)
d = F*m                         # d is a vector in range(A), computed via the foo! method
mul!(d, F, m)                   # equivalent in-place version of the prvious line
A = jacobian(F, rand(domain(A)) # A is the Jacobian of F, a linear operator
```

In addition, same methods that were applied to a jet can be applied to `Jets` operators: `domain`, `range`, `eltype`, `shape`, `size`, `state`, `state!`, `close`. Finally, note that given a linear operator, we can recover the corresponding matrix.
```julia
using Jets, JetPackTransforms
A = JopFft(JetSpace(Float64,5))
B = convert(Array, A)
```

## Operator compositions
Jot operators can be combined in various ways. In this section we consider operator compositions. Operators are composed using `∘` which can be typed into your favorite text editor using unicode. Note that editors such as emacs, vim, atom, vscode, and JupyterLab support using LaTeX. So, typing `\circ` followed by **TAB** will produce `∘`.

**Example of operator compositions**  
```julia
using Pkg
Pkg.add("Jets","JetPack")
using Jets, JetPack
A₁ = JopDiagonal(rand(10))
A₂ = JopDiagonal(rand(10))
A₃ = rand(10,10)
A = A₃ ∘ A₂ ∘ A₁
m = rand(domain(A))
A * m ≈ A₃ * (A₂ * (A₁ * m)) # true
```
Notice that `A₃` is a Julia matrix rather than a Jet operator.

## Operator linear combinations
Operators can be built from linear combinations of operators,
```julia
using Pkg
Pkg.add("Jets","JetPack")
using Jets, JetPack
A₁ = JopDiagonal(rand(10))
A₂ = JopDiagonal(rand(10))
A₃ = rand(10,10)
A = 1.0*A₁ - 2.0*A₂ + 3.0*A₃
m = rand(domain(A))
A*m ≈ 1.0*(A₁*m) - 2.0*(A₂*m) + 3.0*(A₃*m) # true
```

## Block operators, block spaces and block vectors
Jet operators can be combined into block operators which are exactly analogous to block matrices. The domain and ranges of a block operator are of type `JetBSpace` and such that vectors in that space are block vectors of type `BlockArray`. In order to construct a block operator, we use the `@blockop` macro. For example:
```julia
using Pkg
Pkg.add("Jets","JetPack")
using Jets, JetPack
A = @blockop [JopDiagonal(rand(10)) for irow=1:2, icol=1:3]
```
In the above code listing, `A` is a block operator with 2 row-blocks and 3 column-blocks. Given a block operator, we can query for the number of blocks as well as retrieve individual blocks:
```julia
A₁₂ = getblock(A, 1, 2)
nb = nblocks(A)
nrowblocks = nblocks(A, 1)
ncolblocks = nblocks(A, 2)
```
We can form block vectors in the domain and range of `A`. Moreover, once we have formed a block vector, we can access individual blocks. For example,
```julia
d = rand(range(A))
m = rand(domain(A))

nblocks(d)
nblocks(m)

d₂ = getblock(d, 2) # this is not a copy, it is a reference to the second block of d
m₁ = getblock(m, 1)

setblock!(d, 2, rand(size(d₂)))
```
We can reshape Julia Array's into block arrays. For example,
```julia
_d = rand(eltype(range(A)), size(range(A)))
d = reshape(_d, range(A))
```
Since `BlockArrays` extend Julia's `AbstractArray` and broadcasting interfaces, most of the functionality of a Julia `Array` is also available for `BlockArray`'s.

## Vectorized operators
There are libraries that assume that the vectors in the model and data space are backed by one dimensional arrays.  To help with this, Jets provides
a `vec` method that returns an operator with one dimensional arrays backing the domain and range.  As an example, we show how to compose Jets with
the `lsqr` method in the IterativeSolvers package.
```julia
using Jets, JetPack, IterativeSolvers

A = JopDct(Float64, 128, 64)
d = rand(range(A))
m = reshape(lsqr(vec(A), vec(d)), range(A))
A*m ≈ d # true
```
Note that for the case that the domain and range are already backed by one dimensional arrays, `vec` is a no-op.  Further, note that a block array
is a one dimensional array.  

## Creating a new Jet (Developers) 
To build a new jet, provide the function that maps from the domain to the range, its linearization and a default state. We will show three examples: 1) linear operator, 2) self-adjoint linear operator, 3) nonlinear operator.

### Linear operator
```julia
using Jets
MyLinearJet_df!(d, m; A, kwargs...) = mul!(d,A,m)
MyLinearJet_df′!(m, d; A, kwargs...) = mul!(m,A',d)
function MyLinearJet()
    JopLn(dom = JetSpace(Float64,2), rng = JetSpace(Float64,2), 
        df! = MyLinearJet_df!, df′! = MyLinearJet_df′!, s=(A=rand(2,2),))
end
```

### Self-adjoint linear operator
```julia
using Jets
MySelfAdjointJet_df!(d, m; A, kwargs...) = mul!(d,A,m)
function MySelfAdjointJet()
    B = rand(2,2)
    JopLn(dom = JetSpace(Float64,2), rng = JetSpace(Float64,2), 
        df! = MySelfAdjointJet_df!, s = (A=B'*B,))
end
```

### Nonlinear operator
```julia
using Jets
MyNonLinearJet_f!(d, m; a, kwargs...) = d .= x.^a
MyNonLinearJet_df!(d, m; mₒ, a, kwargs...) = d . = a * mₒ.^(a-1) .* m
function MyNonLinearJet()
    JopNl(dom = JetSpace(Float64,2), rng = JetSpace(Float64,2), 
        f! = MyNonLinearJet_f!, df! = MyNonLinearJet_df!, s = (a=2.0,))
end
```
