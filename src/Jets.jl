module Jets

using CRC32c, LinearAlgebra, Random

abstract type JetAbstractSpace{T,N} end

"""
    eltype(R)

Return the element type of the space `R::JetAbstractSpace`
"""
Base.eltype(R::JetAbstractSpace{T}) where {T} = T
Base.ndims(R::JetAbstractSpace{T,N}) where {T,N} = N
Base.eltype(R::Type{JetAbstractSpace{T}}) where {T} = T
Base.eltype(R::Type{JetAbstractSpace{T,N}}) where {T,N} = T

"""
    length(R)

Return the number of elements of the space `R::JetAbstractSpace`
"""
Base.length(R::JetAbstractSpace) = prod(size(R))

"""
    size(R,i)

    Return the shape of R::JetAbstractSpace along dimension i.
"""
Base.size(R::JetAbstractSpace, i) = size(R)[i]

"""
    reshape(x, R)

Reshapes the array `x` to be consistent with the shape of the space `R::JetAbstractSpace`
"""
Base.reshape(x::AbstractArray, R::JetAbstractSpace) = reshape(x, size(R))

struct JetSpace{T,N} <: JetAbstractSpace{T,N}
    n::NTuple{N,Int}
end

"""
    JetSpace(T, n)

Construct and return a JetSpace of type `T` and size `n`

# Example creating a 1D space of type Float64 and size (100,)
```
R1 = JetSpace(Float64, 100)
```

# Example creating a 2D space of type Float32 and size (5, 100)
```
R2 = JetSpace(Float32, 5, 100)
```
"""
JetSpace(_T::Type{T}, n::Vararg{Int,N}) where {T,N} = JetSpace{T,N}(n)
JetSpace(_T::Type{T}, n::NTuple{N,Int}) where {T,N} = JetSpace{T,N}(n)

"""
    size(R)

Return the size of the `Jets` space `R`, as a scalar in the case of singly-dimensional space, 
or as a tuple in the case of multi-dimensional space.
"""
Base.size(R::JetSpace) = R.n

Base.eltype(R::Type{JetSpace{T,N}}) where {T,N} = T
Base.eltype(R::Type{JetSpace{T}}) where {T} = T

@doc """
    Array(R)

Return an array of the type and size defined by the `Jets` space `R`, with values uninitialized.
"""
Array

@doc """
    ones(R)

Return an array of the type and size defined by the `Jets` space `R`, filled with `eltype(R)(1)`.
"""
ones

@doc """
    rand(R)

Return an array of the type and size defined by the `Jets` space `R`, filled with random values.
"""
rand

@doc """
    zeros(R)

Return an array of the type and size defined by the `Jets` space `R`, filled with `eltype(R)(0)`.
"""
zeros

for f in (:ones, :rand, :zeros)
    @eval (Base.$f)(R::JetSpace{T,N}) where {T,N} = ($f)(T,size(R))::Array{T,N}
end
Base.Array(R::JetSpace{T,N}) where {T,N} = Array{T,N}(undef, size(R))

"""
    randperm(R)

Return a list of random linear indices over the length of the space `R`, useful for 
selecting a random subset of a multi-dimensional image, for example.  
"""
Random.randperm(R::JetAbstractSpace, k::Int) = sort(randperm(length(R))[1:k])

jet_missing(m) = error("not implemented")

mutable struct Jet{D<:JetAbstractSpace, R<:JetAbstractSpace, F<:Function, DF<:Function, DF′<:Function, U<:Function, M<:AbstractArray, S<:NamedTuple}
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
    Jet(dom, rng, f!, df!, df'!, upstate!, s)

Return a `Jet` with domain `dom::JetSpace` and range `rng::JetSpace`, with forward mapping 
`f!`, linearized forward mapping `df!`, linearized adjoint mapping `df'!`, Jacobian state
modification function `upstate!`, and state `s`.

A jet describes a function and its linearization at some point in its domain. A jet of the 
smooth function `f` carries information about the values of the function `f(x)`, and the 
differential `d f(x)`. We use jets to define the linearization of nonlinear functions at 
specified point `x`, and provide access to the nonlinear forward map, the linear forward 
map at `x`, and the linear adjoint map at `x`.   

The maps `f!`, `df!`, and `df'!` are initialized to `jet_missing` and will throw an error 
if used without being properly set. 

See also: examples of nonlinear operators in JetPack.jl.
"""
function Jet(; dom, rng, f! = jet_missing, df! = jet_missing, df′! = jet_missing, upstate! = (m,s) -> nothing, s = NamedTuple())
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

Return a `JopNl` with `Jet` constructed from keyword arguments `kwargs`.
"""
JopNl(; kwargs...) = JopNl(Jet(; kwargs...))

struct JopLn{T<:Jet} <: Jop{T}
    jet::T
end

JopLn(jet::Jet, mₒ::AbstractArray) = JopLn(point!(jet, mₒ))

"""
    JopLn(; kwargs ...)
Return a `JopLn` with `Jet` constructed from keyword arguments `kwargs`.
"""
JopLn(;kwargs...) = JopLn(Jet(;kwargs...))

JopLn(A::JopLn) = A
JopLn(F::JopNl) = JopLn(jet(F))

Base.copy(jet::Jet, copymₒ=true) = Jet(jet.dom, jet.rng, jet.f!, jet.df!, jet.df′!, jet.upstate!, copymₒ ? copy(jet.mₒ) : jet.mₒ, deepcopy(jet.s))
Base.copy(F::JopNl, copymₒ=true) = JopNl(copy(jet(F), copymₒ))
Base.copy(A::JopLn, copymₒ=true) = JopLn(copy(jet(A), copymₒ))

struct JopAdjoint{J<:Jet,T<:Jop{J}} <: Jop{J}
    op::T
end

Base.copy(A::JopAdjoint, copymₒ=true) = JopAdjoint(copy(A.op, copymₒ))

JopLn(A::JopAdjoint) = A

"""
    domain(A) 

Return the domain of `A::Union{Jet, Jop, AbstractMatrix}`, which inherits from `JetAbstractSpace`.
"""
domain(jet::Jet) = jet.dom

"""
    range(A) 

Return the range of `A::Union{Jet, Jop, AbstractMatrix}`, which inherits from `JetAbstractSpace`.
"""
Base.range(jet::Jet) = jet.rng

"""
    eltype(jet::Jet) 

Return the type promotion of `eltype(range(jet))` and `eltype(domain(jet))`, using
`Base.promote_type`.
"""
Base.eltype(jet::Jet) = promote_type(eltype(domain(jet)), eltype(range(jet)))

"""
    state(jet::Jet) 

Return the state of the jet `jet::Jet` as a `NamedTuple`.
"""
state(jet::Jet) = jet.s
state(jet::Jet{D,R,F}, key) where {D,R,F<:Function} = jet.s[key]

"""
    state(jet::Jet, s) 

Updates and merges the state of the jet `jet::Jet` with `s`. 
"""
state!(jet, s) = begin jet.s = merge(jet.s, s); jet end

"""
    perfstat(jet)

Return a `Dictionary` with performance information for the maps in the jet. This is a 
method that can be implemented by the author of an operator.
"""
perfstat(jet::T) where {D,R,F<:Function,T<:Jet{D,R,F}} = nothing

"""
    point(A) 

Return the linearization point (model vector) `m₀` associated with `A::Union{Jet, JopLn, JopAdjoint}`.
"""
point(jet::Jet) = jet.mₒ

Base.close(j::Jet{D,R,F}) where {D,R,F<:Function} = false

"""
    point!(jet, mₒ) 

Update the linearization point (model vector) for `jet::Jet` to model vector `m₀`.
"""
function point!(jet::Jet, mₒ::AbstractArray)
    jet.mₒ = mₒ
    jet.upstate!(mₒ, state(jet))
    jet
end

"""
    jet(A) 

Return the `Jet` associated with `A::Jop`.
"""
jet(A::Jop) = A.jet
jet(A::JopAdjoint) = jet(A.op)

"""
    eltype(A::Jop) 

Return the type for the `Jet` associated with the Jets operator `A::Jop`.
"""
Base.eltype(A::Jop) = eltype(jet(A))

point(A::JopLn) = point(jet(A))
point(A::JopAdjoint) = point(jet(A.op))

"""
    state(A) 

Return the state for the `Jet` associated with the Jets operator `A::Union{Jop`.
"""
state(A::Jop) = state(jet(A))
state(A::Jop, key) = state(jet(A), key)

"""
    state!(A::Jop) 

Updates and merges the state for the `Jet` associated with the Jets operator `A::Jop` with `S`.
"""
state!(A::Jop, s) = state!(jet(A), s)

"""
    perfstat(A::Jop)

Return a `Dictionary` with performance information for the Jet operator `A::Jop`.
"""    
perfstat(A::Jop) = perfstat(jet(A))

Base.close(A::Jop) = close(jet(A))

domain(A::Jop) = domain(jet(A))
domain(A::JopAdjoint) = range(A.op)
domain(A::AbstractMatrix{T}) where {T} = JetSpace(T, size(A,2))

Base.range(A::Jop) = range(jet(A))
Base.range(A::JopAdjoint) = domain(A.op)
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

With no arguments, return `(shape(range(A)), shape(domain(A)))`.

With `i` specified, return `shape(range(A))` for `i = 1` and return `shape(domain(A))` for `i = 2`.
"""
shape(A::Union{Jet,Jop}) = (shape(A, 1), shape(A, 2))
shape(A::AbstractMatrix,i) = (size(A, i),)
shape(A::AbstractMatrix) = ((size(A, 1),), (size(A, 2),))

"""
    size(A[, i]) 

Return the size of the range and domain of `A::Union{Jet,Jop}`. 

With no arguments, return `(size(range(A)), size(domain(A)))`.

With `i` specified, return `size(range(A))` for `i = 1` and return `size(domain(A))` for `i = 2`.
"""
Base.size(A::Union{Jet,Jop}, i) = prod(shape(A, i))
Base.size(A::Union{Jet,Jop}) = (size(A, 1), size(A, 2))

"""
    jacobian!(F, m₀) 

Return the jacobian of `F::Union{Jet, JopNl, JopLn, AbstractMatrix}` at the point `m₀`. 
The linearization shares the underlying `Jet` with `F`. 
Note that for linear operators `A::Union{JopLn, AbstractMatrix}`, return `A`.  
"""
jacobian!(jet::Jet, mₒ::AbstractArray) = JopLn(jet, mₒ)
jacobian!(F::JopNl, mₒ::AbstractArray) = jacobian!(jet(F), mₒ)
jacobian!(A::Union{JopLn,AbstractMatrix}, mₒ::AbstractArray) = A

"""
    jacobian(F, m₀) 

Return the jacobian of `F::Union{Jet, JopNl, JopLn, AbstractMatrix}` at the point `m₀`. 
The linearization has a new underlying `Jet`, not shared with `F`.
Note that for linear operators `A::Union{JopLn, AbstractMatrix}`, return `A`.  
"""
jacobian(F::Union{Jet,Jop}, mₒ::AbstractArray) = jacobian!(copy(F, false), copy(mₒ))
jacobian(A::AbstractMatrix, mₒ::AbstractArray) = copy(A)

"""
    adjoint(A) 

Return the adjoint of the `Jets` operator `A::Union{JopLn, JopAdjoint}`. 
`A:JopLn` will return a `JopAdjoint`, and `A:JopAdjoint` will return `JopLn`.
"""
Base.adjoint(A::JopLn) = JopAdjoint(A)
Base.adjoint(A::JopAdjoint) = A.op

"""
    mul!(d, F, m) 

Applies the forward nonlinear map `F::Union{JopNl, JopLn, JopAdjoint}` at the model vector `m` 
and places the result in the data vector `d`. 
"""
LinearAlgebra.mul!(d::AbstractArray, F::JopNl, m::AbstractArray) = f!(d, jet(F), m; state(F)...)
LinearAlgebra.mul!(d::AbstractArray, A::JopLn, m::AbstractArray) = df!(d, jet(A), m; mₒ=point(A), state(A)...)
LinearAlgebra.mul!(m::AbstractArray, A::JopAdjoint{J,T}, d::AbstractArray) where {J<:Jet,T<:JopLn} = 
    df′!(m, jet(A), d; mₒ=point(A), state(A)...)

"""
    :*(A::Jop, m) 

Apply the map `F::Jop` at the model vector `m` and return the result. 
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

`_T` is the type, usually `Complex{Float32}` or `Complex{Float64}`

`n` is a tuple that defines the dimensionality of the space

`M` is a tuple that defines which dimensions are symmetric. Note that currently only a single symmetric dimension is supported by the API.

`F` is a function that maps indices for the symmetric dimension, described below.

An example requiring a `JetSSpace` is the Fourier transform: the Fourier transform of a real vector is in 
a complex space with Hermittian symmetry. Only the positive frequencies are needed, and the spectrum at
negative frequencies is the Hermittian conjugate of the spectrum at the corresponding positive frequencies: 
`S(-f) = conj(S(f)`. For this example the map `F` is a function that returns the multi-dimensional index 
of `f` when given the multi-dimensional index of `-f`. 

See also: `JopFft` in the `JetPackTransforms` package.
"""
JetSSpace(_T::Type{T}, n::NTuple{N,Int}, M::NTuple{N,Int}, map::F) where {T,N,F} = 
    JetSSpace{T,N,F}(n, M, map)

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

for f in (:ones, :rand, :zeros)
    @eval (Base.$f)(R::JetSSpace{T,N,F}) where {T,N,F} = SymmetricArray(($f)(T,R.M), R.n, R.map)::SymmetricArray{T,N,F}
end
Base.Array(R::JetSSpace{T,N,F}) where {T,N,F} = SymmetricArray{T,N,F}(Array{T,N}(undef, R.M), R.n, R.map)

#
# composition, f ∘ g
#
JetComposite(ops) = Jet(dom = domain(ops[end]), rng = range(ops[1]), f! = JetComposite_f!, df! = JetComposite_df!, df′! = JetComposite_df′!, s = (ops=ops,))

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

Construct and return the composition of the two `Jets` operators `A₁::Jop` and `A₂::Jop` as `A₂ ∘ A₁`. 
Note that when applying the composition operator, operators are applied in order from right to left: 
first `A₁` and then `A₂`.
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
# TODO: figure out how to use @doc for both :+ and :-
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

Base.size(R::JetBSpace) = (R.indices[end][end],)
Base.eltype(R::Type{JetBSpace{T,S}}) where {T,S} = T
Base.eltype(R::Type{JetBSpace{T}}) where {T} = T

"""
    indices(R, iblock)

Return the linear indices associated with block `iblock` in the `Jets` block space `R`. 

# Example of block operator with 2 row-blocks and 3 column-blocks.
The call to `indices(domain(A), 1)` in the example below will return the linear indices for the 
domain of the block operator `A`: `1:10`.

```
using Pkg
Pkg.add("Jets", "JetPack")
using Jets, JetPack
A = @blockop [JopDiagonal(rand(10)) for irow=1:2, icol=1:3]
indices(domain(A), 1)
```
"""
indices(R::JetBSpace, iblock::Integer) = R.indices[iblock]

"""
    space(R, iblock)

Return the `Jets` space associated with block `iblock` in the `Jets` block space `R`.

# Example of block operator with 2 row-blocks and 3 column-blocks.
The call to `space(domain(A), 1)` in the example below will return the 1st domain block:
`JetSpace{Float64,1}((10,))`. 

```
using Pkg
Pkg.add("Jets", "JetPack")
using Jets, JetPack
A = @blockop [JopDiagonal(rand(10)) for irow=1:2, icol=1:3]
space(domain(A), 1)
```
"""
space(R::JetBSpace, iblock::Integer) = R.spaces[iblock]

"""
    nblocks(R)

Return the number of blocks in the `Jets` block space `R`.

# Example of block operator with 2 row-blocks and 3 column-blocks.
The call to `nblocks(domain(A))` in the example below will return 3, the number of columns
in the block operator.

```
using Pkg
Pkg.add("Jets", "JetPack")
using Jets, JetPack
A = @blockop [JopDiagonal(rand(10)) for irow=1:2, icol=1:3]
space(domain(A), 1)
```
"""
nblocks(R::JetBSpace) = length(R.spaces)
nblocks(R::JetAbstractSpace) = 1

struct BlockArray{T,A<:AbstractArray{T}} <: AbstractArray{T,1}
    arrays::Vector{A}
    indices::Vector{UnitRange{Int}}
end

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

JopZeroBlock(dom::JetAbstractSpace, rng::JetAbstractSpace) = JopLn(df! = JopZeroBlock_df!, dom = dom, rng = rng)
JopZeroBlock_df!(d, m; kwargs...) = d .= 0

Base.iszero(jet::Jet{D,R,typeof(JopZeroBlock_df!)}) where {D<:JetAbstractSpace,R<:JetAbstractSpace} = true
Base.iszero(jet::Jet) = false
Base.iszero(A::Jop) = iszero(jet(A))

macro blockop(ex)
    :(JopBlock($(esc(ex))))
end

"""
    @blockop(ex, kwargs)

This macro will construct and return a `Jets` block operator, a combination of Jet operators 
exactly analogous to block matrices. The domain and range of block operators are of 
type `JetBSpace`, and vectors in these spaces are of type `BlockArray`. 

`ex` is an 1D or 2D array of `Jets` operators. 

# Example of block operator with 1 row-blocks and 3 column-blocks.

```
using Pkg
Pkg.add("Jets", "JetPack")
using Jets, JetPack
A = @blockop [JopDiagonal(rand(10)) for icol=1:3]
```

# Example of block operator with 2 row-blocks and 3 column-blocks.

```
using Pkg
Pkg.add("Jets", "JetPack")
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

nblocks(jet::Jet) = (nblocks(range(jet)), nblocks(domain(jet)))
nblocks(jet::Jet, i) = i == 1 ? nblocks(range(jet)) : nblocks(domain(jet))
nblocks(A::Jop) = (nblocks(range(A)), nblocks(domain(A)))
nblocks(A::Jop, i) = i == 1 ? nblocks(range(A)) : nblocks(domain(A))

getblock(jet::Jet{D,R,typeof(JetBlock_f!)}, i, j) where {D,R} = state(jet).ops[i,j]
getblock(A::JopLn{T}, i, j) where {T<:Jet} = JopLn(getblock(jet(A), i, j))
getblock(A::JopNl{T}, i, j) where {T<:Jet} = getblock(jet(A), i, j)
getblock(A::T, i, j) where {J<:Jet,T<:JopAdjoint{J}} = getblock(A.op, j, i)'
getblock(::Type{JopNl}, A::Jop{T}, i, j) where {T<:Jet} = getblock(jet(A), i, j)::JopNl
getblock(::Type{JopLn}, A::Jop{T}, i, j) where {T<:Jet} = JopLn(getblock(jet(A), i, j))

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

Base.reshape(x::AbstractArray, R::JetBSpace) = BlockArray([view(x, R.indices[i]) for i=1:length(R.indices)], R.indices)

function Base.close(j::Jet{D,R,typeof(JetBlock_f!)}) where {D,R}
    ops = state(j).ops
    close.(ops)
    nothing
end

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
    dot_product_test(A, m, d; mmask, dmask)

Compute and return the left and right hand sides of the *dot product test*. 

This test is typically defined with random vectors in the domain `m` and range `d` as the 
approximate equality shown below.

<d, A m> ≈ <Aᵀ d, m> 

Here `Aᵀ` is the conjugate transpose or adjoint of `A`, and `<x, y>` is the inner product of 
vectors `x` and `y`. The left and right hand sides of the dot product test are expected to be 
equivalent close to machine precision for operator `A` with correct implementation. If the 
equality does not hold this can indicate a problem with the implementation of the operator `A`.

This function provides the optional named arguments `mmask` and `dmask` which are vectors in the 
domaain and range of `A` that are applied via elementwise multiplication to mask the vectors 
`m` and `d` before applying of the operator, as shown below. Here we use `∘` to represent 
the Hadamard product (elementwise multiplication) of two vectors.

<dmask ∘ d, A (mmask ∘ m)> ≈ <Aᵀ (dmask ∘ d), mmask ∘ m> 

You can test the relative accuracy of the operator with this relation for the left hand side `lhs` and 
right hand side `rhs` returned by this function: 

@assert |lhs - rhs| / |lhs + rhs| < ϵ 

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
    δm ./= maximum(abs, δm)

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
