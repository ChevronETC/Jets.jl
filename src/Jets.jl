module Jets

using CRC32c, LinearAlgebra, Random

abstract type JetAbstractSpace{T,N} end

"""
    eltype(R)

Return the element type of the space `R::JetAbstractSpace`.
"""
Base.eltype(R::JetAbstractSpace{T}) where {T} = T
Base.eltype(R::Type{JetAbstractSpace{T,N}}) where {T,N} = T
Base.eltype(R::Type{JetAbstractSpace{T}}) where {T} = T
Base.ndims(R::JetAbstractSpace{T,N}) where {T,N} = N

"""
    length(R)

Return the dimension the space `R::JetAbstractSpace`
"""
Base.length(R::JetAbstractSpace) = prod(size(R))

"""
    size(R[,i])

Return the shape of the array associated to the Jet space R::JetAbstractSpace.
If `i` is specifid, then returns the length along the ith array dimension.
"""
Base.size(R::JetAbstractSpace, i) = size(R)[i]

"""
    reshape(x, R)

Returns an array that is consistent with the shape of the space `R::JetAbstractSpace`, and shares
memory with `x`.
"""
Base.reshape(x::AbstractArray, R::JetAbstractSpace) = reshape(x, size(R))

struct JetSpace{T,N} <: JetAbstractSpace{T,N}
    n::NTuple{N,Int}
end

"""
    JetSpace(T, n)

Construct and return a JetSpace of type `T` and size `n`

# Examples
Create a 100 dimension space with array elelment type Float64 and array size (100,)
```
R1 = JetSpace(Float64, 100)
```

Create a 100 dimension space with array element type Float32 and array size (5, 20)
```
R2 = JetSpace(Float32, 5, 20)
```
"""
JetSpace(_T::Type{T}, n::Vararg{Int,N}) where {T,N} = JetSpace{T,N}(n)
JetSpace(_T::Type{T}, n::NTuple{N,Int}) where {T,N} = JetSpace{T,N}(n)

Base.size(R::JetSpace) = R.n
Base.eltype(R::Type{JetSpace{T,N}}) where {T,N} = T
Base.eltype(R::Type{JetSpace{T}}) where {T} = T
Base.vec(R::JetSpace) = JetSpace(eltype(R), length(R))

@doc """
    Array(R)

Construct an uninitialized array of the type and size defined by `R::JetsAbstractSpace`.
"""
Array

@doc """
    ones(R)

Construct an array of the type and size defined by `R::JetAbstractSpace{T}` and filled with `one(T)`.
"""
ones

@doc """
    rand(R)

Construct an array of the type and size defined by the `R::JetAbstractSpace`, and filled with random values.
"""
rand

@doc """
    zeros(R)

Construct an array of the type and size defined by `R::JetAbstractSpace{T}` and filled with `zero(T)`.
"""
zeros

for f in (:ones, :rand, :zeros)
    @eval (Base.$f)(R::JetSpace{T,N}) where {T,N} = ($f)(T,size(R))::Array{T,N}
end
Base.Array(R::JetSpace{T,N}) where {T,N} = Array{T,N}(undef, size(R))

"""
    randperm(R)

Construct a list of random linear indices over the dimensions of `R::JetAbstractSpace`.  The list
is useful for selecting a random subset of a multi-dimensional image.

# Example
```julia
using Jets
R = JetSpace(Float64, 10, 2)
x = rand(R)
y = x[randperm(R)[1:10]] # get 10 elements at random from x
```
"""
Random.randperm(R::JetAbstractSpace, k::Int) = sort(randperm(length(R))[1:k])

space(x::AbstractArray{T,N}) where {T,N} = JetSpace{T,N}(size(x))

jet_missing(m) = error("not implemented")

mutable struct Jet{D<:JetAbstractSpace,R<:JetAbstractSpace,F<:Function,DF<:Function,DF′<:Function,U<:Function,M<:AbstractArray,S<:NamedTuple}
    dom::D
    rng::R
    f!::F
    df!::DF
    df′!::DF′
    upstate!::U
    mₒ::M
    s::S
end

"""
    Jet(;dom, rng, f!, df!, df′!, upstate!, s)

Return a `Jet` with domain `dom::JetAbstractSpace`, range `rng::JetSAbstractpace`, with
forward mapping  `f!::Function`, linearized forward mapping `df!::Function`, linearized adjoint
mapping `df′!::Function`, Jacobian state modification function `upstate!::Function`, and state
`s::NamedTuple`.

A jet describes a function `f!` and its linearization (forward `df!, and adjoint `df′!``) about
a point.

If one of `f!` or `df!` is specified, and `df′!` is not, then we assume that `f!=df!=df′!`. This means
that the operator is linear and self-adjoint.

If `f!` and `df!` are sepecified, bug `df′!` is not, then we assume that `df′!=df!`.  This means
that the operator is nonlinear and self-adjoint.

# Example
Consider a nonlinear mapping with a self-adjoint linearization ``f(x)=x^2``
```julia
using Jets
g!(m) = m.^2
dg!(δm; mₒ) = @. 2*mₒ*δm
jet = Jet(;dom=JetSpace(Float64,2), rng=JetSpace(Float64,2), f! = g!, df! = dg!)
```
"""
function Jet(;
        dom,
        rng,
        f! = jet_missing,
        df! = jet_missing,
        df′! = jet_missing,
        upstate! = (m,s) -> nothing,
        s = NamedTuple())
    if isa(f!, typeof(jet_missing)) && isa(df!, typeof(jet_missing))
        error("must set at-least one of f! and df!")
    end
    if isa(f!, typeof(jet_missing))
        f! = df!
    end
    if isa(df′!, typeof(jet_missing))
        df′! = df!
    end
    Jet(dom, rng, f!, df!, df′!, upstate!, Array{eltype(dom)}(undef, ntuple(i->0,ndims(dom))), s)
end

f!(d, jet::Jet, m; kwargs...) = jet.f!(d, m; kwargs...)
df!(d, jet::Jet, m; kwargs...) = jet.df!(d, m; kwargs...)
df′!(m, jet::Jet, d; kwargs...) = jet.df′!(m, d; kwargs...)

abstract type Jop{T<:Jet} end

struct JopNl{T<:Jet} <: Jop{T}
    jet::T
end

"""
    JopNl(; kwargs ...)

Construct a `JopNl` with `Jet` constructed from keyword arguments `kwargs`.
This is equivalent to `JopNl(Jet(;kwargs...))`.  Please see `Jet` for more
information.
"""
JopNl(;kwargs...) = JopNl(Jet(;kwargs...))

struct JopLn{T<:Jet} <: Jop{T}
    jet::T
end
JopLn(jet::Jet, mₒ::AbstractArray) = JopLn(point!(jet, mₒ))

"""
    JopLn(; kwargs ...)

Construct a `JopLn` with `Jet` constructed from keyword arguments `kwargs`.
This is equivalent to `JopLn(Jet(;kwargs...))`.  Please see `Jet` for more
information.
"""
JopLn(;kwargs...) = JopLn(Jet(;kwargs...))

JopLn(A::JopLn) = A
JopLn(F::JopNl) = JopLn(jet(F))

struct JopAdjoint{J<:Jet,T<:Jop{J}} <: Jop{J}
    op::T
end

Base.copy(jet::Jet, copymₒ=true) = Jet(jet.dom, jet.rng, jet.f!, jet.df!, jet.df′!, jet.upstate!, copymₒ ? copy(jet.mₒ) : jet.mₒ, deepcopy(jet.s))
Base.copy(A::JopLn, copymₒ=true) = JopLn(copy(jet(A), copymₒ))
Base.copy(A::JopAdjoint, copymₒ=true) = JopAdjoint(copy(A.op, copymₒ))
Base.copy(F::JopNl, copymₒ=true) = JopNl(copy(jet(F), copymₒ))

JopLn(A::JopAdjoint) = A

"""
    R = domain(A)

Return `R::JetAbstractSpace`, which is the domain of `A::Union{Jet, Jop, AbstractMatrix}`.
"""
domain(jet::Jet) = jet.dom

"""
    R = range(A)

Return `R::JetAbstractSpace`, which is the range of `A::Union{Jet, Jop, AbstractMatrix}`.
"""
Base.range(jet::Jet) = jet.rng

"""
    eltype(A::Union{Jet,Jop,JopAdjoint})

Return the element type of `A`.
"""
Base.eltype(jet::Jet) = promote_type(eltype(domain(jet)), eltype(range(jet)))

"""
    state(A::Union{Jet,Jop,JopAdjoint}[, key])

If `key::Symbol` is specified, then return the state corresponding to `key`.
Otherwise, return the state of A as a `NamedTuple`.
"""
state(jet::Jet) = jet.s
state(jet::Jet{D,R,F}, key) where {D,R,F<:Function} = jet.s[key]

"""
    state!(A::Union{Jet,Jop,JopAdjoint}, s)

Updates and merges the state of the `A` with `s`.
"""
state!(jet, s) = begin jet.s = merge(jet.s, s); jet end

"""
    perfstat(A)

Return a `Dictionary` with performance information for A::Union{Jet,Jop,JopAdjoint}. the
`perfstat(jet::Jet)` method that can be implemented by the author of an operator to track
performance metrics.
"""
perfstat(jet::T) where {D,R,F<:Function,T<:Jet{D,R,F}} = nothing

"""
    point(F)

Return the linearization point (model vector) `mₒ` associated with `F::Union{Jet, JopLn, JopAdjoint}`.
"""
point(jet::Jet) = jet.mₒ

Base.close(j::Jet{D,R,F}) where {D,R,F<:Function} = false

"""
    point!(F, mₒ)

Update the linearization point (model vector) for `F::Union{Jet, JopLn, JopAdjoint}` to model vector `mₒ`.
"""
function point!(jet::Jet, mₒ::AbstractArray)
    jet.mₒ = mₒ
    jet.upstate!(mₒ, state(jet))
    jet
end

"""
    jet(A)

Return the `Jet` associated with `A::Union{Jop, JopAdjoint}`.
"""
jet(A::Jop) = A.jet
jet(A::JopAdjoint) = jet(A.op)
Base.eltype(A::Jop) = eltype(jet(A))
point(A::JopLn) = point(jet(A))
point(A::JopAdjoint) = point(jet(A.op))
state(A::Jop) = state(jet(A))
state(A::Jop, key) = state(jet(A), key)
state!(A::Jop, s) = state!(jet(A), s)
perfstat(A::Jop) = perfstat(jet(A))
Base.close(A::Jop) = close(jet(A))

domain(A::Jop) = domain(jet(A))
Base.range(A::Jop) = range(jet(A))

domain(A::JopAdjoint) = range(A.op)
Base.range(A::JopAdjoint) = domain(A.op)

domain(A::AbstractMatrix{T}) where {T} = JetSpace(T, size(A,2))
Base.range(A::AbstractMatrix{T}) where {T} = JetSpace(T, size(A,1))

function shape(A::Union{Jet,Jop}, i)
    if i == 1
        return size(range(A))
    end
    size(domain(A))
end

"""
    shape(A[, i])

Return the shape of the range and domain of `A::Union{Jet, Jop, AbstractMatrix}`.
With no arguments, return `(size(range(A)), size(domain(A)))`.  With `i` specified,
return `size(range(A))` for `i = 1` and return `size(domain(A))` otherwise.
"""
shape(A::Union{Jet,Jop}) = (shape(A, 1), shape(A, 2))

shape(A::AbstractMatrix,i) = (size(A, i),)
shape(A::AbstractMatrix) = ((size(A, 1),), (size(A, 2),))

"""
    size(A[, i])

Return the size of the range and domain of `A::Union{Jet,Jop}`.  With no arguments, return
`(length(range(A)), length(domain(A)))`.  With `i` specified, return `length(range(A))` for `i = 1`
and return `length(domain(A))` otherwise.
"""
Base.size(A::Union{Jet,Jop}, i) = prod(shape(A, i))
Base.size(A::Union{Jet,Jop}) = (size(A, 1), size(A, 2))

"""
    jacobian!(F, m₀)

Return the jacobian of `F::Union{Jet, Jop, AbstractMatrix}` at the
linearization point `m₀`. The jacobian shares the underlying `Jet` with `F`.
This means that if the jacobian may mutate `F`.
"""
jacobian!(jet::Jet, mₒ::AbstractArray) = JopLn(jet, mₒ)
jacobian!(F::JopNl, mₒ::AbstractArray) = jacobian!(jet(F), mₒ)
jacobian!(A::Union{JopLn,AbstractMatrix}, mₒ::AbstractArray) = A

"""
    jacobian(F, m₀)

Return the jacobian of `F::Union{Jet, Jop, AbstractMatrix}` at the point `m₀`.
The linearization constructs a new underlying `Jet`.
"""
jacobian(F::Union{Jet,Jop}, mₒ::AbstractArray) = jacobian!(copy(F, false), copy(mₒ))
jacobian(A::AbstractMatrix, mₒ::AbstractArray) = copy(A)

"""
    adjoint(A::Union{JopLn, JopAdjoint})

Return the adjoint of A.
"""
Base.adjoint(A::JopLn) = JopAdjoint(A)
Base.adjoint(A::JopAdjoint) = A.op

"""
    mul!(d, F, m)

In place version of `d=F*m` where F is a Jets linear (e.g. `JopLn`) or nonlinear (`JopNl`) operator.
"""
LinearAlgebra.mul!(d::AbstractArray, F::JopNl, m::AbstractArray) = f!(d, jet(F), m; state(F)...)
LinearAlgebra.mul!(d::AbstractArray, A::JopLn, m::AbstractArray) = df!(d, jet(A), m; mₒ=point(A), state(A)...)
LinearAlgebra.mul!(m::AbstractArray, A::JopAdjoint{J,T}, d::AbstractArray) where {J<:Jet,T<:JopLn} = df′!(m, jet(A), d; mₒ=point(A), state(A)...)

"""
    :*(F, m)

Constructs `F*m` where F is a Jets linear (e.g. `JopLn`) or nonlinear (`JopNl`) operator.
"""
Base.:*(A::Jop, m::AbstractArray) = mul!(zeros(range(A)), A, m)

Base.show(io::IO, A::JopLn) = show(io, "Jet linear operator, $(size(domain(A))) → $(size(range(A)))")
Base.show(io::IO, A::JopAdjoint) = show(io, "Jet adjoint operator, $(size(domain(A))) → $(size(range(A)))")
Base.show(io::IO, F::JopNl) = show(io, "Jet nonlinear operator, $(size(domain(F))) → $(size(range(F)))")

#
# Symmetric spaces / arrays
#
struct JetSSpace{T,N,F<:Function} <: JetAbstractSpace{T,N}
    n::NTuple{N,Int}
    M::NTuple{N,Int}
    map::F
end

"""
    JetSSpace(_T, n, M, map::F)

Construct and return a symmetric space `JetSSpace`.

# parameters
* `_T` is the type, usually `Complex{Float32}` or `Complex{Float64}`.
* `n` is a tuple that defines the dimensionality of the space.
* `M` is a tuple that defines which dimensions are symmetric. Note that currently only a single symmetric dimension is supported by the API.
* `F` is a function that maps indices for the symmetric dimension, described below.

An example requiring a `JetSSpace` is the Fourier transform: the Fourier transform of a real vector is in
a complex space with Hermittian symmetry. Only the positive frequencies are needed, and the spectrum at
negative frequencies is the Hermittian conjugate of the spectrum at the corresponding positive frequencies:
`S(-f) = conj(S(f)`. For this example the map `F` is a function that returns the multi-dimensional index
of `f` when given the multi-dimensional index of `-f`.

See also: `JopFft` in the `JetPackTransforms` package.
"""
JetSSpace(_T::Type{T}, n::NTuple{N,Int}, M::NTuple{N,Int}, map::F) where {T,N,F} = JetSSpace{T,N,F}(n, M, map)

Base.size(R::JetSSpace) = R.n
Base.eltype(R::Type{JetSSpace{T,N,F}}) where {T,N,F} = T
Base.eltype(R::Type{JetSSpace{T,N}}) where {T,N} = T
Base.eltype(R::Type{JetSSpace{T}}) where {T} = T
symspace() = nothing

struct SymmetricArray{T,N,F<:Function} <: AbstractArray{T,N}
    A::Array{T,N}
    n::NTuple{N,Int}
    map::F
end

space(x::SymmetricArray{T,N,F}) where {T,N,F} = JetSSpace{T,N,F}(x.n, size(x.A), x.map)

Base.parent(A::SymmetricArray) = A.A

# SymmetricArray array interface implementation <--
Base.IndexStyle(::Type{T}) where {T<:SymmetricArray} = IndexCartesian()
Base.size(A::SymmetricArray) = A.n

function Base.getindex(A::SymmetricArray{T,N}, I::Vararg{Int,N}) where {T,N}
    for idim = 1:ndims(A)
        if I[idim] > size(A.A, idim)
            return conj(A.A[A.map(I)])
        end
    end
    A.A[CartesianIndex(I)]
end

function Base.getindex(A::SymmetricArray, i::Int)
    I = CartesianIndices(size(A))[i]
    getindex(A, I.I...)
end

function Base.setindex!(A::SymmetricArray{T,N}, v, I::Vararg{Int,N}) where {T,N}
    for idim = 1:ndims(A)
        if I[idim] > size(A.A, idim)
            A.A[A.map(I)] = conj(v)
            return A.A[A.map(I)]
        end
    end
    A.A[CartesianIndex(I)] = v
end

function Base.setindex!(A::SymmetricArray, v, i::Int)
    I = CartesianIndices(size(A))[i]
    setindex!(A, v, I.I...)
end

Base.similar(A::SymmetricArray, ::Type{T}) where {T<:Complex} = SymmetricArray(similar(A.A), A.n, A.map)
Base.similar(A::SymmetricArray, ::Type{T}) where {T<:Real} = Array{T}(undef, size(A))
Base.similar(A::SymmetricArray{T}) where {T} = similar(A, T)
# -->

# SymmetricArray broadcasting interface implementation --<
Base.BroadcastStyle(::Type{<:SymmetricArray}) = Broadcast.ArrayStyle{SymmetricArray}()

Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{SymmetricArray}}, ::Type{T}) where {T} = similar(find_symmetricarray(bc), T)

find_symmetricarray(bc::Broadcast.Broadcasted) = find_symmetricarray(bc.args)
find_symmetricarray(args::Tuple) = find_symmetricarray(find_symmetricarray(args[1]), Base.tail(args))
find_symmetricarray(x) = x
find_symmetricarray(a::SymmetricArray, rest) = a
find_symmetricarray(::Any, rest) = find_symmetricarray(rest)

get_symmetricarray_parent(bc::Broadcast.Broadcasted, ::Type{S}) where {S} = Broadcast.Broadcasted{S}(bc.f, map(arg->get_symmetricarray_parent(arg, S), bc.args))
get_symmetricarray_parent(A::SymmetricArray, ::Type{<:Any}) = parent(A)
get_symmetricarray_parent(A, ::Type{<:Any}) = A

function Base.copyto!(dest::SymmetricArray{T,N}, bc::Broadcast.Broadcasted{Nothing}) where {T,N}
    S = Broadcast.DefaultArrayStyle{N}
    copyto!(parent(dest), get_symmetricarray_parent(bc, S))
    dest
end
# -->

for f in (:ones, :rand, :zeros)
    @eval (Base.$f)(R::JetSSpace{T,N,F}) where {T,N,F} = SymmetricArray(($f)(T,R.M), R.n, R.map)::SymmetricArray{T,N,F}
end
Base.Array(R::JetSSpace{T,N,F}) where {T,N,F} = SymmetricArray{T,N,F}(Array{T,N}(undef, R.M), R.n, R.map)

#
# composition, f ∘ g
#

JetComposite(ops) = Jet(f! = JetComposite_f!, df! = JetComposite_df!, df′! = JetComposite_df′!, dom = domain(ops[end]), rng = range(ops[1]), s = (ops=ops,))

function JetComposite_f!(d::T, m; ops, kwargs...) where {T<:AbstractArray}
    g = mapreduce(i->(_m->mul!(zeros(range(ops[i])), ops[i], _m)), ∘, 1:length(ops))
    d .= g(m)
    d::T
end

function JetComposite_df!(d::T, m; ops, kwargs...) where {T<:AbstractArray}
    dg = mapreduce(i->(_m->mul!(zeros(range(JopLn(ops[i]))), JopLn(ops[i]), _m)), ∘, 1:length(ops))
    d .= dg(m)
    d::T
end

function JetComposite_df′!(m::T, d; ops, kwargs...) where {T<:AbstractArray}
    dg′ = mapreduce(i->(_d->mul!(zeros(domain(JopLn(ops[i]))), (JopLn(ops[i]))', _d)), ∘, length(ops):-1:1)
    m .= dg′(d)
    m::T
end

jops_comp(op::Jop) = (op,)
jops_comp(op::JopLn{J}) where {D,R,J<:Jet{D,R,typeof(JetComposite_f!)}} = state(jet(op)).ops
jops_comp(op::JopNl{J}) where {D,R,J<:Jet{D,R,typeof(JetComposite_f!)}} = state(jet(op)).ops

function jops_comp(op::JopAdjoint{J,T}) where {D,R,J<:Jet{D,R,typeof(JetComposite_f!)},T<:JopLn{J}}
    ops = state(op.op).ops
    n = length(ops)
    ntuple(i->JopAdjoint(ops[n-i+1]), n)
end

"""
    :∘(A₂, A₁)

Construct the composition of the two `Jets` operators.  Note that when applying the composition
operator, operators are applied in order from right to left: first `A₁` and then `A₂`.

# Example
```julia
using Jets
dg!(d,m;mₒ) = @. d = 2*m
A₁ = JopLn(Jet(;dom=JetSpace(Float64,2), rng=JetSpace(Float64,2), df! = dg!))
A₂ = JopLn(Jet(;dom=JetSpace(Float64,2), rng=JetSpace(Float64,2), df! = dg!))
C = A₂ ∘ A₁
m = rand(domain(C))
d = C * m
```
"""
Base.:∘(A₂::Union{JopAdjoint,JopLn}, A₁::Union{JopAdjoint,JopLn}) = JopLn(JetComposite((jops_comp(A₂)..., jops_comp(A₁)...)))
Base.:∘(A₂::Jop, A₁::Jop) = JopNl(JetComposite((jops_comp(A₂)..., jops_comp(A₁)...)))
Base.:∘(A₂::AbstractMatrix, A₁::AbstractMatrix) = A₂*A₁

_matmul_df!(d, m; A, kwargs...) = mul!(d, A, m)
_matmul_df′!(m, d; A, kwargs...) = mul!(m, adjoint(A), d)
Base.:∘(A₂::Jop, A₁::AbstractMatrix) = A₂ ∘ JopLn(;dom = domain(A₁), rng = range(A₁), df! = _matmul_df!, df′! = _matmul_df′!, s=(A=A₁,))
Base.:∘(A₂::AbstractMatrix, A₁::Jop) = JopLn(;dom = domain(A₂), rng = range(A₂), df! = _matmul_df!, df′! = _matmul_df′!, s=(A=A₂,)) ∘ A₁

function point!(j::Jet{D,R,typeof(JetComposite_f!)}, mₒ::AbstractArray) where {D<:JetAbstractSpace,R<:JetAbstractSpace}
    j.mₒ = mₒ
    ops = state(j).ops
    _m = copy(mₒ)
    for i = length(ops):-1:1
        point!(jet(ops[i]), _m)
        if i > 1
            _m = mul!(zeros(range(ops[i])), ops[i], _m)
        end
    end
    j
end

function Base.close(j::Jet{D,R,typeof(JetComposite_f!)}) where {D,R}
    ops = state(j).ops
    close.(ops)
    nothing
end

function perfstat(j::Jet{D,R,typeof(JetComposite_f!)}) where {D,R}
    ops = state(j).ops
    s = nothing
    for op in ops
        s = perfstat(op)
        s == nothing || break
    end
    s
end

function state(j::Jet{D,R,typeof(JetComposite_f!)}, key) where {D,R}
    key ∈ keys(state(j)) && (return state(j)[key])

    haskey = 0
    local _op
    for op in state(j).ops
        if key ∈ keys(state(op))
            haskey += 1
            _op = op
        end
    end

    haskey == 0 && error("key $key does not exist in the state of the composite operator")
    haskey > 1 && error("ambiguous: key $key exists in more than one operator in the composition")

    state(_op, key)
end

#
# Composition, f ± g
#
JetSum(ops, sgns) = Jet(f! = JetSum_f!, df! = JetSum_df!, df′! = JetSum_df′!, dom = domain(ops[1]), rng = range(ops[1]), s = (ops=ops, sgns=sgns))

function JetSum_f!(d, m; ops, sgns, kwargs...)
    d .= 0
    _d = zeros(range(ops[1]))
    for i = 1:length(ops)
        broadcast!(sgns[i], d, d, mul!(_d, ops[i], m))
    end
    d
end

function JetSum_df!(d, m; ops, sgns, kwargs...)
    d .= 0
    _d = zeros(range(ops[1]))
    for i = 1:length(ops)
        broadcast!(sgns[i], d, d, mul!(_d, JopLn(ops[i]), m))
    end
    d
end

function JetSum_df′!(m, d; ops, sgns, kwargs...)
    m .= 0
    _m = zeros(domain(ops[1]))
    for i = 1:length(ops)
        broadcast!(sgns[i], m, m, mul!(_m, JopLn(ops[i])', d))
    end
    m
end

jops_sum(op::Jop) = (op,)
jops_sum(op::JopLn{J}) where {D,R,J<:Jet{D,R,typeof(JetSum_f!)}} = state(jet(op)).ops
jops_sum(op::JopNl{J}) where {D,R,J<:Jet{D,R,typeof(JetSum_f!)}} = state(jet(op)).ops

function jops_sum(op::JopAdjoint{J,T}) where {D,R,J<:Jet{D,R,typeof(JetSum_f!)},T<:JopLn{J}}
    ops = state(op.op).ops
    n = length(ops)
    ntuple(i->JopAdjoint(ops[i]), n)
end

function flipsgn(sgn, sgnnew)
    typeof(sgnnew) == typeof(+) && (return sgn)
    typeof(sgnnew) == typeof(-) && typeof(sgn) == typeof(-) && (return +)
    typeof(sgnnew) == typeof(-) && typeof(sgn) == typeof(+) && (return -)
end

sgns(op::Jop, r) = (r,)
sgns(op::JopLn{J}, r) where {D,R,J<:Jet{D,R,typeof(JetSum_f!)}} = ntuple(i->flipsgn(state(op).sgns[i],r), length(state(op).sgns))
sgns(op::JopNl{J}, r) where {D,R,J<:Jet{D,R,typeof(JetSum_f!)}} = ntuple(i->flipsgn(state(op).sgns[i],r), length(state(op).sgns))
sgns(op::JopAdjoint{J,T}, r) where {D,R,J<:Jet{D,R,typeof(JetSum_f!)},T<:JopLn{J}} = ntuple(i->flipsgn(state(op).sgns[i],r), length(state(op.op).sgns))

"""
    :+(A₂, A₁)

Construct and return the linear combination of the two `Jets` operators `A₁::Jop` and `A₂::Jop`.
Note that `A₁` and `A₂` must have consistent (same size and type) domains and ranges.

# Example
```
A = 1.0*A₁ - 2.0*A₂ + 3.0*A₃
```
"""
Base.:+(A₂::Union{JopAdjoint,JopLn}, A₁::Union{JopAdjoint,JopLn}) = JopLn(JetSum((jops_sum(A₂)..., jops_sum(A₁)...), (sgns(A₂,+)..., sgns(A₁,+)...)))
Base.:+(A₂::Jop, A₁::Jop) = JopNl(JetSum((jops_sum(A₂)..., jops_sum(A₁)...), (sgns(A₂,+)..., sgns(A₁,+)...)))
Base.:+(A₂::Jop, A₁::AbstractMatrix) = A₂ + JopLn(;dom = domain(A₁), rng = range(A₁), df! = _matmul_df!, df′! = _matmul_df′!, s=(A=A₁,))
Base.:+(A₂::AbstractMatrix, A₁::Jop) = JopLn(;dom = domain(A₂), rng = range(A₂), df! = _matmul_df!, df′! = _matmul_df′!, s=(A=A₂,)) + A₁

"""
    :-(A₂, A₁)

Construct and return the linear combination of the two `Jets` operators `A₁::Jop` and `A₂::Jop`.
Note that `A₁` and `A₂` must have consistent (same size and type) domains and ranges.

# Example
```
A = 1.0*A₁ - 2.0*A₂ + 3.0*A₃
```
"""
Base.:-(A₂::Union{JopAdjoint,JopLn}, A₁::Union{JopAdjoint,JopLn}) = JopLn(JetSum((jops_sum(A₂)..., jops_sum(A₁)...), (sgns(A₂,+)..., sgns(A₁,-)...)))
Base.:-(A₂::Jop, A₁::Jop) = JopNl(JetSum((jops_sum(A₂)..., jops_sum(A₁)...), (sgns(A₂,+)..., sgns(A₁,-)...)))
Base.:-(A₂::Jop, A₁::AbstractMatrix) = A₂ - JopLn(;dom = domain(A₁), rng = range(A₁), df! = _matmul_df!, df′! = _matmul_df′!, s=(A=A₁,))
Base.:-(A₂::AbstractMatrix, A₁::Jop) = JopLn(;dom = domain(A₂), rng = range(A₂), df! = _matmul_df!, df′! = _matmul_df′!, s=(A=A₂,)) - A₁

function point!(j::Jet{D,R,typeof(JetSum_f!)}, mₒ::AbstractArray) where {D<:JetAbstractSpace,R<:JetAbstractSpace}
    for op in state(j).ops
        point!(jet(op), mₒ)
    end
    j
end

function Base.close(j::Jet{D,R,typeof(JetSum_f!)}) where {D,R}
    ops = state(j).ops
    close.(ops)
    nothing
end

function perfstat(j::Jet{D,R,typeof(JetSum_f!)}) where {D,R}
    ops = state(j).ops
    s = nothing
    for op in ops
        s = perfstat(op)
        s == nothing || break
    end
    s
end

#
# Block operator
#
struct JetBSpace{T,S<:JetAbstractSpace} <: JetAbstractSpace{T,1}
    spaces::Vector{S}
    indices::Vector{UnitRange{Int}}
    function JetBSpace(spaces)
        S = mapreduce(typeof, promote_type, spaces)
        T = eltype(S)
        indices = [0:0 for i=1:length(spaces)]
        stop = 0
        for i = 1:length(indices)
            start = stop + 1
            stop = start + length(spaces[i]) - 1
            indices[i] = start:stop
        end
        new{T,S}(spaces, indices)
    end
end

Base.:(==)(x::JetBSpace, y::JetBSpace) = x.spaces == y.spaces && x.indices == y.indices

Base.size(R::JetBSpace) = (R.indices[end][end],)
Base.eltype(R::Type{JetBSpace{T,S}}) where {T,S} = T
Base.eltype(R::Type{JetBSpace{T}}) where {T} = T
Base.vec(R::JetBSpace) = R

"""
    indices(R, iblock)

Return the linear indices associated with block `iblock` in the `Jets` block space
`R::JetBSpace`.

# Example
Consider a block operator with 2 row-blocks and 3 column-blocks.  We can use `indices`
to determine the elements of the vector that are associatd with the first block of its
domain:
```
using Pkg
pkg"add Jets JetPack"
using Jets, JetPack
A = @blockop [JopDiagonal(rand(10)) for irow=1:2, icol=1:3]
indices(domain(A), 1) # returns indices 1:10
```
"""
indices(R::JetBSpace, iblock::Integer) = R.indices[iblock]

"""
    space(R, iblock)

Return the `Jets` sub-space associated with block `iblock` in the `Jets` block space
`R::JetBSpace`.

# Example
Consider a block operator with 2 row-blocks and 3 column-blocks.  We can use `space` to
determine the sub-space associated with the first block of its domain:
```
using Pkg
pkg"add Jets JetPack"
using Jets, JetPack
A = @blockop [JopDiagonal(rand(10)) for irow=1:2, icol=1:3]
space(domain(A), 1) # JetSpace(Float64,10)
```
"""
space(R::JetBSpace, iblock::Integer) = R.spaces[iblock]

"""
    nblocks(R)

Return the number of blocks in the `Jets` block space `R::JetBSpace`.
"""
nblocks(R::JetBSpace) = length(R.spaces)
nblocks(R::JetAbstractSpace) = 1

struct BlockArray{T,A<:AbstractArray{T}} <: AbstractArray{T,1}
    arrays::Vector{A}
    indices::Vector{UnitRange{Int}}
end

space(x::BlockArray) = JetBSpace([space(_x) for _x in x.arrays])

# BlockArray array interface implementation <--
Base.IndexStyle(::Type{T}) where {T<:BlockArray} = IndexLinear()
Base.size(A::BlockArray) = (A.indices[end][end],)

function Base.getindex(A::BlockArray, i::Int)
    j = findfirst(rng->i∈rng, A.indices)::Int
    A.arrays[j][i-A.indices[j][1]+1]
end
function Base.setindex!(A::BlockArray, v, i::Int)
    j = findfirst(rng->i∈rng, A.indices)::Int
    A.arrays[j][i-A.indices[j][1]+1] = v
end

Base.similar(A::BlockArray, ::Type{T}) where {T} = BlockArray([similar(A.arrays[i], T) for i=1:length(A.arrays)], A.indices)
Base.similar(A::BlockArray{T}) where {T} = similar(A, T)
Base.similar(x::BlockArray, ::Type{T}, n::Integer) where {T} = length(x) == n ? similar(x, T) : Array{T}(undef, n)
Base.similar(x::BlockArray, ::Type{T}, dims::Tuple{Int}) where {T} = similar(x, T, dims[1])

function LinearAlgebra.norm(x::BlockArray{T}, p::Real=2) where {T}
    if p == Inf
        mapreduce(_x->norm(_x,p), max, x.arrays)
    elseif p == -Inf
        mapreduce(_x->norm(_x,p), min, x.arrays)
    elseif p == 1
        mapreduce(_x->norm(_x,p), +, x.arrays)
    elseif p == 0
        mapreduce(_x->norm(_x,p), +, x.arrays)
    else
        _T = float(real(T))
        _p = _T(p)
        mapreduce(_x->norm(_x,p)^_p, +, x.arrays)^(one(_T)/_p)
    end
end

function LinearAlgebra.dot(x::BlockArray{T}, y::BlockArray{T}) where {T}
    a = zero(T)
    for i = 1:length(x.arrays)
        a += dot(x.arrays[i], y.arrays[i])
    end
    a
end

indices(x::BlockArray, i) = x.indices[i]

nblocks(x::BlockArray) = length(x.indices)

function Base.convert(::Type{Array}, x::BlockArray{T}) where {T}
    _x = Vector{T}(undef, length(x))
    for i = 1:length(x.indices)
        _x[indices(x,i)] .= vec(x.arrays[i])
    end
    _x
end

function Base.extrema(x::BlockArray{T}) where {T}
    mn,mx = extrema(x.arrays[1])
    for i = 2:length(x.arrays)
        _mn, _mx = extrema(x.arrays[i])
        _mn < mn && (mn = _mn)
        _mx > mx && (mx = _mx)
    end
    mn,mx
end

function Base.fill!(x::BlockArray, a)
    for i = 1:length(x.arrays)
        fill!(x.arrays[i], a)
    end
    x
end
# -->

# BlockArray broadcasting implementation --<
struct BlockArrayStyle <: Broadcast.AbstractArrayStyle{1} end
Base.BroadcastStyle(::Type{<:BlockArray}) = BlockArrayStyle()
BlockArrayStyle(::Val{1}) = BlockArrayStyle()

Base.similar(bc::Broadcast.Broadcasted{BlockArrayStyle}, ::Type{T}) where {T} = similar(find_blockarray(bc), T)
find_blockarray(bc::Broadcast.Broadcasted) = find_blockarray(bc.args)
find_blockarray(args::Tuple) = find_blockarray(find_blockarray(args[1]), Base.tail(args))
find_blockarray(x) = x
find_blockarray(a::BlockArray, rest) = a
find_blockarray(::Any, rest) = find_blockarray(rest)

getblock(bc::Broadcast.Broadcasted, ::Type{S}, iblock, indices) where {S} = Broadcast.Broadcasted{S}(bc.f, map(arg->getblock(arg, S, iblock, indices), bc.args))
getblock(A::BlockArray, ::Type{<:Any}, iblock, indices) = getblock(A, iblock)
getblock(A::AbstractArray, ::Type{<:Any}, iblock, indices) = A[indices]
getblock(A, ::Type{<:Any}, iblock, indices) = A

function Base.copyto!(dest::BlockArray{T,<:AbstractArray{T,N}}, bc::Broadcast.Broadcasted{BlockArrayStyle}) where {T,N}
    S = Broadcast.DefaultArrayStyle{N}
    for iblock = 1:nblocks(dest)
        copyto!(getblock(dest, iblock), getblock(bc, S, iblock, dest.indices[iblock]))
    end
    dest
end
# -->

getblock(x::BlockArray, iblock) = x.arrays[iblock]
getblock!(x::BlockArray, iblock, xblock::AbstractArray) = xblock .= x.arrays[iblock]
setblock!(x::BlockArray, iblock, xblock) = x.arrays[iblock] .= xblock

getblock(x::AbstractArray, iblock) = x
getblock!(x::AbstractArray, iblock, xblock::AbstractArray) = xblock .= x
setblock!(x::AbstractArray, iblock, xblock) = x .= xblock

for f in (:Array, :ones, :rand, :zeros)
    @eval (Base.$f)(R::JetBSpace{T,S}) where {T,S<:JetAbstractSpace} = BlockArray([($f)(space(R, i)) for i=1:length(R.spaces)], R.indices)
end

function JetBlock(ops::AbstractMatrix{T}; dadom=false, kwargs...) where {T<:Jop}
    dom = (size(ops,2) == 1 && !dadom) ? domain(ops[1,1]) : JetBSpace([domain(ops[1,i]) for i=1:size(ops,2)])
    rng = JetBSpace([range(ops[i,1]) for i=1:size(ops,1)])
    Jet(f! = JetBlock_f!, df! = JetBlock_df!, df′! = JetBlock_df′!, dom = dom, rng = rng, s = (ops=ops,dom=dom,rng=rng))
end
JopBlock(ops::AbstractMatrix{T}; kwargs...) where {T<:Union{JopLn,JopAdjoint}} = JopLn(JetBlock(ops; kwargs...))
JopBlock(ops::AbstractMatrix{T}; kwargs...) where {T<:Jop} = JopNl(JetBlock(ops; kwargs...))
JopBlock(ops::AbstractVector{T}; kwargs...) where {T<:Jop} = JopBlock(reshape(ops, length(ops), 1); kwargs...)

"""
    JopZeroBlock(dom, rng)

Construct a Jets operator that is equivalent to a matrix of zeros, and that maps from `dom::JetAbstractSpace`
to `rng::JetAbstractSpace`.  This can be useful when forming block operators that contain zero blocks.
"""
JopZeroBlock(dom::JetAbstractSpace, rng::JetAbstractSpace) = JopLn(df! = JopZeroBlock_df!, dom = dom, rng = rng)
JopZeroBlock_df!(d, m; kwargs...) = d .= 0

"""
    iszero(A::Union{Jet, Jop})

Return true if `A` was constructed via `JopZeroBlock`.
"""
Base.iszero(jet::Jet{D,R,typeof(JopZeroBlock_df!)}) where {D<:JetAbstractSpace,R<:JetAbstractSpace} = true
Base.iszero(jet::Jet) = false
Base.iszero(A::Jop) = iszero(jet(A))

macro blockop(ex)
    :(JopBlock($(esc(ex))))
end

"""
    @blockop(ex, kwargs)

Construct a `Jets` block operator, a combination of Jet operators
analogous to a block matrix.

# Examples
example with 1 row-block and 3 column-blocks:
```julia
using Pkg
pkg"add Jets JetPack"
using Jets, JetPack
A = @blockop [JopDiagonal(rand(10)) for icol=1:3]
```

example with 2 row-blocks and 3 column-blocks.
```
using Pkg
pkg"add Jets JetPack"
using Jets, JetPack
A = @blockop [JopDiagonal(rand(10)) for irow=1:2, icol=1:3]
```
"""
macro blockop(ex, kw)
    :(JopBlock($(esc(ex)); $(esc(kw))))
end

macro blockop(ex, kw1, kw2)
    :(JopBlock($(esc(ex)); $(esc(kw1)), $(esc(kw2))))
end

function JetBlock_f!(d::AbstractArray, m::AbstractArray; ops, dom, rng, kwargs...)
    local dtmp
    if size(ops, 2) > 1
        dtmp = zeros(range(ops[1,1]))
    end
    for iblkrow = 1:size(ops, 1)
        _d = getblock(d, iblkrow)
        if size(ops, 2) > 1
            dtmp = size(dtmp) == size(range(ops[iblkrow,1])) ? dtmp : zeros(range(ops[iblkrow,1]))
        end
        for iblkcol = 1:size(ops, 2)
            _m = getblock(m, iblkcol)
            if size(ops, 2) > 1
                _d .+= mul!(dtmp, ops[iblkrow,iblkcol], _m)
            else
                mul!(_d, ops[iblkrow,iblkcol], m)
            end
        end
    end
    d
end

function JetBlock_df!(d::AbstractArray, m::AbstractArray; ops, dom, rng, kwargs...)
    local dtmp
    if size(ops, 2) > 1
        dtmp = zeros(range(ops[1,1]))
    end
    for iblkrow = 1:size(ops, 1)
        _d = getblock(d, iblkrow)
        if size(ops, 2) > 1
            dtmp = size(dtmp) == size(range(ops[iblkrow,1])) ? dtmp : zeros(range(ops[iblkrow,1]))
        end
        for iblkcol = 1:size(ops, 2)
            _m = getblock(m, iblkcol)
            if !iszero(ops[iblkrow,iblkcol])
                if size(ops, 2) > 1
                    _d .+= mul!(dtmp, JopLn(ops[iblkrow,iblkcol]), _m)
                else
                    mul!(_d, JopLn(ops[iblkrow,iblkcol]), _m)
                end
            end
        end
    end
    d
end

function JetBlock_df′!(m::AbstractArray, d::AbstractArray; ops, dom, rng, kwargs...)
    local mtmp
    if size(ops, 1) > 1
        mtmp = zeros(domain(ops[1,1]))
    end
    for iblkcol = 1:size(ops, 2)
        _m = getblock(m, iblkcol)
        if size(ops, 1) > 1
            _m .= 0
            mtmp = size(mtmp) == size(domain(ops[1,iblkcol])) ? mtmp : zeros(domain(ops[1,iblkcol]))
        end
        for iblkrow = 1:size(ops, 1)
            _d = getblock(d, iblkrow)
            if !iszero(ops[iblkrow,iblkcol])
                if size(ops, 1) > 1
                    _m .+= mul!(mtmp, (JopLn(ops[iblkrow,iblkcol]))', _d)
                else
                    mul!(_m, (JopLn(ops[iblkrow,iblkcol]))', _d)
                end
            end
        end
    end
    m
end

function point!(j::Jet{D,R,typeof(JetBlock_f!)}, mₒ::AbstractArray) where {D<:JetAbstractSpace,R<:JetAbstractSpace}
    ops = state(j).ops
    dom = domain(j)
    for icol = 1:size(ops, 2), irow = 1:size(ops, 1)
        point!(jet(ops[irow,icol]), getblock(mₒ, icol))
    end
    j
end

"""
    nblocks(A[, i])

Return the number of blocks in the range and domain of the `Jets` block operator `A::Union{Jet, Jop}`.
With `i` specified, return `nblocks(range(jet))` for `i = 1` and return `nblocks(domain(jet))` otherwise.
"""
nblocks(jet::Jet) = (nblocks(range(jet)), nblocks(domain(jet)))
nblocks(jet::Jet, i) = i == 1 ? nblocks(range(jet)) : nblocks(domain(jet))
nblocks(A::Jop) = (nblocks(range(A)), nblocks(domain(A)))
nblocks(A::Jop, i) = i == 1 ? nblocks(range(A)) : nblocks(domain(A))

"""
    getblock(A, i, j)

Return the block of the Jets block operator `A` that corresponds to row block `i`
and column block `j`.
"""
getblock(jet::Jet{D,R,typeof(JetBlock_f!)}, i, j) where {D,R} = state(jet).ops[i,j]
getblock(A::JopLn{T}, i, j) where {T<:Jet} = JopLn(getblock(jet(A), i, j))
getblock(A::JopNl{T}, i, j) where {T<:Jet} = getblock(jet(A), i, j)
getblock(A::T, i, j) where {J<:Jet,T<:JopAdjoint{J}} = getblock(A.op, j, i)'
getblock(::Type{JopNl}, A::Jop{T}, i, j) where {T<:Jet} = getblock(jet(A), i, j)::JopNl
getblock(::Type{JopLn}, A::Jop{T}, i, j) where {T<:Jet} = JopLn(getblock(jet(A), i, j))

"""
    isblockop(A)

Return true if `A` is a Jets block operator.
"""
isblockop(A::Jop{<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetBlock_f!)}}) = true
isblockop(A::Jop) = false

function getblock(jet::Jet{D,R,typeof(JetComposite_f!)}, i, j) where {D,R}
    ops = []
    for op in state(jet).ops
        if isblockop(op)
            push!(ops, getblock(op, i, j))
        else
            push!(ops, op)
        end
    end
    mapreduce(identity, ∘, ops)
end

Base.reshape(x::AbstractArray, R::JetBSpace) = BlockArray([reshape(view(x, R.indices[i]), R.spaces[i]) for i=1:length(R.indices)], R.indices)

# BlockArrays are vectors
function Base.reshape(x::BlockArray, R::JetBSpace)
    length(x) == length(R) || error("dimension mismatch, unable to reshape block array")
    x
end

function Base.close(j::Jet{D,R,typeof(JetBlock_f!)}) where {D,R}
    ops = state(j).ops
    close.(ops)
    nothing
end

#
# Vectorized operator
#
JetVec(op::Jop) = Jet(f! = JetVec_f!, df! = JetVec_df!, df′! = JetVec_df′!, dom = vec(domain(op)), rng = vec(range(op)), s=(op=op,))
JetVec(op::Jop{<:J}) where {T,U,D<:JetAbstractSpace{T,1},R<:JetAbstractSpace{T,1},J<:Jet{D,R}} = op
JopVec(op::Union{JopLn,JopAdjoint}) = JopLn(JetVec(op))
JopVec(op::Jop) = JopNl(JetVec(op))

JetVec_f!(d::T, m::AbstractVector; op, kwargs...) where {T<:AbstractVector} = vec(mul!(reshape(d, range(op)), op, reshape(m, domain(op))))::T
JetVec_df!(d::T, m::AbstractVector; op, kwargs...) where {T<:AbstractVector} = vec(mul!(reshape(d, range(op)), JopLn(op), reshape(m, domain(op))))::T
JetVec_df′!(m::T, d::AbstractVector; op, kwargs...) where {T<:AbstractVector} = vec(mul!(reshape(m, domain(op)), JopLn(op)', reshape(d,range(op))))::T

"""
    B = vec(A)

B is equivelent to A except that its domain and range are "vectorized".  This is
useful when calling algorithms that expect vectors in the domain and range of the
operator.  One example of this is the `lsqr` method in the IterativeSolvers package.

# Example
```julia
using Jets, JetPack, IterativeSolvers

A = JopDiagonal(rand(10,11))
d = rand(range(A))
m = lsqr(vec(A), vec(d))
```
"""
Base.vec(op::Jop) = JopVec(op)

#
# multiply operator by a scalar
#
_constdiag_df!(d, m; a, kwargs...) = d .= a * m
_constdiag_df′!(m, d; a, kwargs...) = m .= conj(a) * d
function Base.:*(a::Number, A::Jop)
    _a = JopLn(dom = domain(A), rng = domain(A), df! = _constdiag_df!, df′! = _constdiag_df′!, s=(a=a,))
    _a ∘ A
end

#
# utilities
#
"""
    convert(Array, A::JopLn)

Convert a linear Jets operator into its equivalent matrix.
"""
function Base.convert(::Type{T}, A::Jop) where {T<:Array}
    m = zeros(domain(A))
    d = zeros(range(A))
    B = zeros(eltype(A), size(A))
    for icol = 1:size(A, 2)
        m .= 0
        d .= 0
        m[icol] = 1
        B[:,icol] .= mul!(d, A, m)[:]
    end
    B
end

"""
    lhs,rhs = dot_product_test(A, m, d; mmask, dmask)

Compute and return the left and right hand sides of the *dot product test*:

`<d,Am> ≈ <Aᴴd,m>`

Here `Aᴴ` is the conjugate transpose or adjoint of `A`, and `<x, y>` denotes the inner
product of vectors `x` and `y`. The left and right hand sides of the dot product test are
expected to be equivalent close to machine precision for operator `A`. If the equality does not
hold this can indicate a problem with the implementation of the operator `A`.

This function provides the optional named arguments `mmask` and `dmask` which are vectors in the
domain and range of `A` that are applied via elementwise multiplication to mask the vectors
`m` and `d` before applying of the operator, as shown below. Here we use `∘` to represent
the Hadamard product (elementwise multiplication) of two vectors.

`<dmask ∘ d, A (mmask ∘ m)> ≈ <Aᵀ (dmask ∘ d), mmask ∘ m>`

You can test the relative accuracy of the operator with this relation for the left hand side `lhs` and
right hand side `rhs` returned by this function: 

`|lhs - rhs| / |lhs + rhs| < ϵ`
"""
function dot_product_test(op::JopLn, m::AbstractArray, d::AbstractArray; mmask=[], dmask=[])
    mmask = length(mmask) == 0 ? ones(domain(op)) : mmask
    dmask = length(dmask) == 0 ? ones(range(op)) : dmask

    ds = op * (mmask .* m)
    ms = op' * (dmask .* d)

    lhs = dot(mmask.*m, ms)
    rhs = dot(ds, dmask.*d)

    if eltype(lhs) <: Complex && eltype(rhs) <: Complex
        return lhs, rhs
    else
        return real(lhs), real(rhs)
    end
end

"""
    μobs, μexp = linearization_test(F, mₒ; μ)

Thest that the jacobian, `J`, of `F` satisfies the Taylor expansion:

`F(m) = F(m_o) + F'(m_o)δm + O(δm^2)`
"""
function linearization_test(F::JopNl, mₒ::AbstractArray;
        μ=[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125], δm=[], mmask=[], dmask = [], seed=Inf)
    mmask = length(mmask) == 0 ? ones(domain(F)) : mmask
    dmask = length(dmask) == 0 ? ones(range(F)) : dmask

    isfinite(seed) && Random.seed!(seed)
    if length(δm) == 0
        δm = mmask .* (-1 .+ 2 .* rand(domain(F)))
    else
        δm .*= mmask
    end

    Fₒ = F*mₒ
    Jₒ = jacobian!(F, mₒ)
    Jₒδm = Jₒ*δm

    μ = convert(Array{eltype(mₒ)}, sort(μ, rev=true))

    μobs = zeros(length(μ)-1)
    μexp = zeros(length(μ)-1)
    ϕ = zeros(length(μ))
    for i = 1:length(μ)
        d_lin  = Fₒ .+ μ[i] .* Jₒδm
        d_non  = F*(mₒ .+ μ[i] .* δm)
        ϕ[i] = norm(dmask .* (d_non .- d_lin))
        if i>1
            μobs[i-1] = ϕ[i-1]/ϕ[i]
            μexp[i-1] = (μ[i-1]/μ[i]).^2
        end
    end
    μobs, μexp
end

"""
    lhs,rhs = linearity_test(A::Jop)

test the the linear Jet operator `A` satisfies the following test
for linearity: 
    
`A(m_1+m_2)=Am_1 + A_m2`
"""
function linearity_test(A::Union{JopLn,JopAdjoint})
    m1 = -1 .+ 2 * rand(domain(A))
    m2 = -1 .+ 2 * rand(domain(A))
    lhs = A*(m1 + m2)
    rhs = A*m1 + A*m2
    lhs, rhs
end

# for hashing models <--
CRC32c.crc32c(m::Array{<:Union{UInt32,Float32,Float64,Complex{Float32},Complex{Float64}}}) = CRC32c.crc32c(unsafe_wrap(Array, convert(Ptr{UInt8}, pointer(m)), length(m)*sizeof(eltype(m)), own=false))
#-->

export Jet, JetAbstractSpace, JetBSpace, JetSpace, JetSSpace, Jop, JopAdjoint, JopLn, JopNl,
    JopZeroBlock, @blockop, domain, getblock, getblock!, dot_product_test, getblock,
    getblock!, indices, jacobian, jacobian!, jet, linearity_test, linearization_test,
    nblocks, perfstat, point, setblock!, shape, space, state, state!, symspace

end
