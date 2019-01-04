#=
NOTES...
* we use an extra field (a named tuple) in JotOp to keep operator state
* how do we do an in-place multiply -- mul!(d,A,m) ... use the state(A.s)?
* I could not figure out how to make the block operator without including spaces in JotOp
* Can we use LinearMaps.jl, AbstractOperators.jl?

TODO...
 * adjoint
 * kron, +, -
 * distributed (memory)
 * distributed (file system)
=#

module JotNew

using LinearAlgebra

struct JotSpace{T,N}
    n::NTuple{N,Int}
end
JotSpace(_T::Type{T}, n::Vararg{Int,N}) where {T,N} = JotSpace{T,N}(n)
JotSpace(_T::Type{T}, n::NTuple{N,Int}) where {T,N} = JotSpace{T,N}(n)

Base.size(R::JotSpace) = R.n
Base.eltype(R::JotSpace{T}) where {T} = T
Base.ndims(R::JotSpace{T,N}) where {T,N} = N
Base.reshape(x::AbstractArray, R::JotSpace) = reshape(x, size(R))

for f in (:ones, :rand, :zeros)
    @eval (Base.$f)(R::JotSpace{T}) where {T} = ($f)(T,R.n)
end

jot_missing(m) = error("not implemented")

struct Jet{F<:Function,DF<:Function,DF′<:Function,D<:JotSpace,R<:JotSpace,S<:NamedTuple}
    f::F
    df::DF
    df′::DF′
    s::S
    dom::D
    rng::R
end
function Jet(;
        f=jot_missing,
        df=jot_missing,
        df′=jot_missing,
        s=NamedTuple(),
        dom,
        rng)
    s = merge(s, (mₒ=Array{eltype(dom)}(undef, ntuple(i->0,ndims(dom))),))
    Jet(f, df, df′, s, dom ,rng)
end

abstract type JotOp end

struct JotOpNl{T<:Jet} <: JotOp
    jet::T
end
JotOpNl(;kwargs...) = JotOpNl(Jet(;merge(kwargs,mₒ=[])...)) # TODO

forward(F::JotOpNl, m) = F.jet.f(m;state(F)...)

struct JotOpLn{T<:Jet} <: JotOp
    jet::T
end
JotOpLn(;kwargs...) = JotOpLn(Jet(;kwargs...))

forward(A::JotOpLn,m) = A.jet.df(m;state(A)...)
adjoint(A::JotOpLn,d) = A.jet.df′(d;state(A)...)

jet(A::JotOp) = A.jet
state(A::JotOp) = jet(A).s

function shape(jet::Jet, i)
    if i == 1
        return size(range(jet))
    end
    size(domain(jet))
end
shape(jet::Jet) = (shape(jet, 1), shape(jet, 2))
shape(op::JotOp, i) = shape(jet(op), i)
shape(op::JotOp) = shape(jet(op))

shape(A::AbstractMatrix,i) = size(A, i)
shape(A::AbstractMatrix) = size(A)

Base.size(jet::Jet, i) = prod(shape(jet, i))
Base.size(jet::Jet) = (size(jet, 1), size(jet, 2))
Base.size(A::JotOp, i) = size(jet(A), i)
Base.size(A::JotOp) = size(jet(A))

domain(jet::Jet) = jet.dom
Base.range(jet::Jet) = jet.rng

domain(A::JotOp) = domain(jet(A))
Base.range(A::JotOp) = range(jet(A))

function jacobian(F::JotOpNl, mₒ::AbstractArray)
    state(F) = merge(state(F), (mₒ=mₒ,))
    JotOpLn(jet(F))
end
jacobian(A::JotOpLn, mₒ::AbstractArray) = A

Base.:*(A::JotOp, m::AbstractArray) = forward(A,m)

#
# compositions
#
struct JotOpComposite{T<:Tuple} <: JotOp
    ops::T
end
Base.:∘(A₂::JotOp, A₁::JotOp) = JotOpComposite((A₁, A₂))
Base.:∘(A₂::JotOp, A₁::JotOpComposite) = JotOpComposite((A₁.ops..., A₂))
Base.:∘(A₂::JotOpComposite, A₁::JotOp) = JotOpComposite((A₁, A₂.ops...))
Base.:∘(A₂::JotOpComposite, A₁::JotOpComposite) = JotOpComposite((A₁.ops..., A₂.ops...))

Base.size(A::JotOpComposite) = (size(A.ops[end], 1), size(A.ops[1], 2))
Base.size(A::JotOpComposite, i) = size(A.ops)[i]
shape(A::JotOpComposite) = (shape(A.ops[end], 1), shape(A.ops[1], 2))
shape(A::JotOpComposite, i) = shape(A.ops)[i]
domain(A::JotOpComposite) = domain(A.ops[end])
Base.range(A::JotOpComposite) = range(A.ops[1])

function Base.:*(A::JotOpComposite, m::AbstractArray)
    f = _m->forward(A.ops[1], _m)
    for i = 2:length(A.ops)
        f = (_m->forward(A.ops[i], _m)) ∘ f
    end
    f(m)
end

function jacobian(A::JotOpComposite, m::AbstractArray)
    ops = []
    m = forward(op.ops[1], m)
    if isa(A.ops[1], JotOpLn)
        push!(ops, op.ops[1])
    else
        push!(ops, jacobian(op.ops[1], m))
    end
    for i = 2:length(op.ops)
        if isa(A.ops[i], JotOpLn)
            push!(ops, op.ops[i])
        else
            push!(ops, jacobian(op.ops[i], m))
        end
        if i < length(op.ops)
            m = forward(op.ops[i], m)
        end
    end
    JotOpComposite((ops...))
end

#
# Block operators
#
struct JotOpBlock{T<:JotOp} <: JotOp
    ops::Matrix{T}
end

Base.hcat(A::JotOp...) = JotOpBlock([A[j] for i=1:1, j=1:length(A)])
Base.vcat(A::JotOp...) = JotOpBlock([A[i] for i=1:length(A), j=1:1])
Base.vect(A::JotOp...) = JotOpBlock([A[i] for i=1:length(A), j=1:1])
Base.hvcat(rows::Tuple{Vararg{Int}}, xs::JotOp...) = JotOpBlock(Base.typed_hvcat(Base.promote_typeof(xs...), rows, xs...))
Base.getindex(A::JotOpBlock, i, j) = A.ops[i,j]

function domain(A::JotOpBlock)
    D = promote_type(ntuple(i->eltype(range(A[i,1])), size(A,1))...)
    Dn = mapreduce(i->size(A[i,1],1), +, 1:size(A,1))
    JotSpace(D,Dn)
end
function Base.range(A::JotOpBlock)
    R = promote_type(ntuple(i->eltype(range(A[1,i])), size(A,2))...)
    Rn = mapreduce(i->size(A[1,i],2), +, 1:size(A,2))
    JotSpace(R,Rn)
end
function shape(A::JotOpBlock, i)
    if i == 1
        return size(range(A))
    end
    size(domain(A))
end
shape(A::JotOpBlock) = (shape(A,1), shape(A,2))
Base.size(A::JotOpBlock, i) = prod(shape(A,i))
Base.size(A::JotOpBlock) = (size(A,1), size(A,2))
nblocks(A::JotOpBlock,i) = size(A.ops,i)
nblocks(A::JotOpBlock) = size(A.ops)

function Base.:*(A::JotOpBlock, m::AbstractArray)
    d = zeros(range(A))
    rngN = 0
    for iblkrow = 1:nblocks(A,1)
        rng1 = rngN + 1
        rngN = rng1 + size(A[iblkrow,1],1) - 1
        _d = @view d[rng1:rngN]

        domN = 0
        for iblkcol = 1:nblocks(A,2)
            dom1 = domN + 1
            domN = dom1 + size(A[1,iblkcol],2) - 1
            _m = @view m[dom1:domN]
            _d .+= forward(A[iblkrow,iblkcol], reshape(_m, domain(A[iblkrow,iblkcol])))
        end
    end
    d
end

forward(A::JotOpBlock, m::AbstractArray) = A*m

function jacobian(F::JotOpBlock, m::AbstractArray)
    domrng = Vector{UnitRange}(undef, nblocks(F, 2))
    domN = 0
    for i = 1:nblocks(F, 2)
        dom1 = domN + 1
        domN = dom1 + size(F[1,iblkcol], 2)
        domrng[i] = dom1:domN
    end
    [jacobian(F[i,j], @view(m[domrng[j]])) for i=1:nblocks(F,1), j=1:nblocks(F,2)]
end

export JotOp, JotSpace, domain, nblocks, jacobian, jet, shape, state

#
# test operators
#

# linear operator
function JotOpDiagonal(d)
    df(m;kwargs...) = kwargs[:diagonal] .* m
    df′(d;kwargs...) = df(d;kwargs...)
    spc = JotSpace(eltype(d),size(d))
    JotOpLn(;df=df, df′=df′, dom=spc, rng=spc, s=(diagonal=d,))
end

# non-linear operator
function JotOpSquare(n)
    f(m;kwargs...) = m.^2
    df(δm;kwargs...) = 2 .* kwargs[:mₒ] .* δm
    spc = JotSpace(Float64, n)
    JotOpNl(;f=f, df=df, df′=df, dom=spc, rng=spc)
end

export JotOpDiagonal, JotOpSquare

end
