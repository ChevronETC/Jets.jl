#=
NOTES...
* we use an extra field (a named tuple) in JotOp to keep operator state
* I could not figure out how to make the block operator without including spaces in JotOp
* Can we use LinearMaps.jl, AbstractOperators.jl, etc.?

TODO...
 * revisit type stability / generated code
   - compoiste operator has unstable mul,mul!
   - block operator has unstable constructor
 * LinearAlgebra has an Adjoint type, can we use it?
 * propagating @inbounds
 * distributed block operator (memory)
 * distributed block operator (file system)
 * fault tolerance
 * kron, +, -
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
Base.eltype(R::Type{JotSpace{T,N}}) where {T,N} = T
Base.eltype(R::Type{JotSpace{T}}) where {T} = T
Base.ndims(R::JotSpace{T,N}) where {T,N} = N
Base.reshape(x::AbstractArray, R::JotSpace) = reshape(x, size(R))

for f in (:ones, :rand, :zeros)
    @eval (Base.$f)(R::JotSpace{T}) where {T} = ($f)(T,R.n)
end

jot_missing(m) = error("not implemented")

mutable struct Jet{D<:JotSpace,R<:JotSpace,F<:Function,DF<:Function,DF′<:Function,S<:NamedTuple}
    dom::D
    rng::R
    f!::F
    df!::DF
    df′!::DF′
    s::S
end
function Jet(;
        dom,
        rng,
        f! = jot_missing,
        df! = jot_missing,
        df′! = jot_missing,
        s = NamedTuple())
    s = merge(s, (mₒ=Array{eltype(dom)}(undef, ntuple(i->0,ndims(dom))),))
    Jet(dom, rng, f!, df!, df′!, s)
end

abstract type JotOp end

struct JotOpNl{T<:Jet} <: JotOp
    jet::T
end
JotOpNl(;kwargs...) = JotOpNl(Jet(;kwargs...))

struct JotOpLn{T<:Jet} <: JotOp
    jet::T
end
JotOpLn(;kwargs...) = JotOpLn(Jet(;kwargs...))

struct JotOpAdjoint{T<:JotOp} <: JotOp
    op::T
end

domain(jet::Jet) = jet.dom
Base.range(jet::Jet) = jet.rng
state(jet::Jet) = jet.s
state!(jet, s) = jet.s = merge(jet.s, s)

jet(A::JotOp) = A.jet
jet(A::JotOpAdjoint) = jet(A.op)
state(A::JotOp) = state(jet(A))
state!(A::JotOp, s) = state!(jet(A), s)

domain(A::JotOp) = domain(jet(A))
Base.range(A::JotOp) = range(jet(A))

domain(A::JotOpAdjoint) = range(A.op)
Base.range(A::JotOpAdjoint) = domain(A.op)

function shape(A::Union{Jet,JotOp}, i)
    if i == 1
        return size(range(A))
    end
    size(domain(A))
end
shape(A::Union{Jet,JotOp}) = (shape(A, 1), shape(A, 2))

shape(A::AbstractMatrix,i) = size(A, i)
shape(A::AbstractMatrix) = size(A)

Base.size(A::Union{Jet,JotOp}, i) = prod(shape(A, i))
Base.size(A::Union{Jet,JotOp}) = (size(A, 1), size(A, 2))

function jacobian(F::JotOpNl, mₒ::AbstractArray)
    state!(F, merge(state(F), (mₒ=mₒ,)))
    JotOpLn(jet(F))
end
jacobian(A::JotOpLn, mₒ::AbstractArray) = A

Base.adjoint(A::JotOpLn) = JotOpAdjoint(A)
Base.adjoint(A::JotOpAdjoint) = A.op

LinearAlgebra.mul!(d::AbstractArray, F::JotOpNl, m::AbstractArray) = jet(F).f!(d, m; state(F)...)
LinearAlgebra.mul!(d::AbstractArray, A::JotOpLn, m::AbstractArray) = jet(A).df!(d, m; state(A)...)
LinearAlgebra.mul!(m::AbstractArray, A::JotOpAdjoint{T}, d::AbstractArray) where {T<:JotOpLn} = jet(A).df′!(m, d; state(A)...)

Base.:*(A::JotOp, m::AbstractArray) = mul!(zeros(range(A)), A, m)

#
# composition
#
struct JotOpComposite{T<:Tuple, D<:JotSpace, R<:JotSpace} <: JotOp
    ops::T
    function JotOpComposite(ops::T) where {T<:Tuple}
        D = typeof(domain(ops[end]))
        R = typeof(range(ops[1]))
        new{T,D,R}(ops)
    end
end
Base.:∘(A₂::JotOp, A₁::JotOp) = JotOpComposite((A₂, A₁))
Base.:∘(A₂::JotOp, A₁::JotOpComposite) = JotOpComposite((A₂, A₁.ops...))
Base.:∘(A₂::JotOpComposite, A₁::JotOp) = JotOpComposite((A₂.ops..., A₁))
Base.:∘(A₂::JotOpComposite, A₁::JotOpComposite) = JotOpComposite((A₂.ops..., A₁.ops...))

domain(A::JotOpComposite) = domain(A.ops[end])
Base.range(A::JotOpComposite) = range(A.ops[1])

function LinearAlgebra.mul!(d::T, A::JotOpComposite, m::AbstractArray) where {T<:AbstractArray}
    f = mapreduce(i->(_m->A.ops[i]*_m), ∘, 1:length(A.ops))
    d .= f(m)
    d::T
end

function LinearAlgebra.mul!(m::T, A::JotOpAdjoint{O}, d::AbstractArray) where {T<:AbstractArray, O<:JotOpComposite}
    f = mapreduce(i->(_d->adjoint(A.op.ops[i])*_d), ∘, length(A.op.ops):-1:1)
    m .= f(d)
    m::T
end

Base.adjoint(A::JotOpComposite) = JotOpAdjoint(A)

function jacobian(A::JotOpComposite, m::AbstractArray)
    ops = []
    for i = 1:length(A.ops)
        push!(ops, jacobian(A.ops[i], m))
        if i < length(A.ops)
            m = A.ops[i]*m
        end
    end
    JotOpComposite((ops...,))
end

#
# Block operator
#
struct JotOpBlock{D<:JotSpace,R<:JotSpace,T<:JotOp} <: JotOp
    ops::Matrix{T}
    function JotOpBlock(ops::Matrix{T}) where {T<:JotOp}
        D = promote_type(ntuple(i->eltype(range(ops[i,1])), size(ops,1))...)
        R = promote_type(ntuple(i->eltype(range(ops[1,i])), size(ops,2))...)
        new{JotSpace{D,1},JotSpace{R,1},T}(ops)
    end
end

struct JotOpZeroBlock{T,N} <: JotOp
    dom::JotSpace{T,N}
    rng::JotSpace{T,N}
end
jacobian(A::JotOpZeroBlock, m) = A

Base.hcat(A::JotOp...) = JotOpBlock([A[j] for i=1:1, j=1:length(A)])
Base.vcat(A::JotOp...) = JotOpBlock([A[i] for i=1:length(A), j=1:1])
Base.vect(A::JotOp...) = JotOpBlock([A[i] for i=1:length(A), j=1:1])
Base.hvcat(rows::Tuple{Vararg{Int}}, xs::JotOp...) = JotOpBlock(Base.typed_hvcat(Base.promote_typeof(xs...), rows, xs...))
Base.getindex(A::JotOpBlock, i, j) = A.ops[i,j]
Base.getindex(A::JotOpAdjoint{T}, i, j) where {T<:JotOpBlock} = A.op.ops[j,i]

function domain(A::JotOpBlock{D,R,T}) where {D,R,T}
    n = mapreduce(i->size(A[1,i],2), +, 1:nblocks(A,2))
    JotSpace(eltype(D),n)
end
function Base.range(A::JotOpBlock{D,R,T}) where {D,R,T}
    n = mapreduce(i->size(A[i,1],1), +, 1:nblocks(A,1))
    JotSpace(eltype(R),n)
end

nblocks(A::JotOpBlock) = size(A.ops)
nblocks(A::JotOpBlock, i) = size(A.ops, i)

function LinearAlgebra.mul!(d::AbstractArray, A::JotOpBlock, m::AbstractArray)
    rngN = 0
    for iblkrow = 1:nblocks(A,1)
        rng1 = rngN + 1
        rngN = rng1 + size(A[iblkrow,1],1) - 1
        _d = @view d[rng1:rngN]

        domN = 0
        for iblkcol = 1:nblocks(A,2)
            dom1 = domN + 1
            domN = dom1 + size(A[1,iblkcol],2) - 1
            if !isa(A[iblkrow,iblkcol], JotOpZeroBlock)
                _m = @view m[dom1:domN]
                _d .+= A[iblkrow,iblkcol] * reshape(_m, domain(A[iblkrow,iblkcol]))
            end
        end
    end
    d
end

function LinearAlgebra.mul!(m::AbstractArray, B::JotOpAdjoint{T}, d::AbstractArray) where {T<:JotOpBlock}
    A = B.op
    domN = 0
    for iblkcol = 1:nblocks(A,2)
        dom1 = domN + 1
        domN = dom1 + size(A[1,iblkcol],2) - 1
        _m = @view m[dom1:domN]

        rngN = 0
        for iblkrow = 1:nblocks(A,1)
            rng1 = rngN + 1
            rngN = rng1 + size(A[iblkrow,1],1) - 1
            if !isa(A[iblkrow,iblkcol], JotOpZeroBlock)
                _d = @view d[rng1:rngN]
                _m .+= adjoint(A[iblkrow,iblkcol]) * reshape(_d, range(A[iblkrow,iblkcol]))
            end
        end
    end
    m
end

Base.adjoint(A::JotOpBlock{D,R,T}) where {D,R,T} = JotOpAdjoint(A)

function jacobian(F::JotOpBlock, m::AbstractArray)
    domrng = Vector{UnitRange}(undef, nblocks(F, 2))
    domN = 0
    for i = 1:nblocks(F, 2)
        dom1 = domN + 1
        domN = dom1 + size(F[1,i], 2) - 1
        domrng[i] = dom1:domN
    end
    JotOpBlock([jacobian(F[i,j], @view(m[domrng[j]])) for i=1:nblocks(F,1), j=1:nblocks(F,2)])
end

function blockrange(A::JotOpBlock, iblock)
    i1 = iblock == 1 ? 1 : mapreduce(i->size(A[i,1],1), +, 1:(iblock-1)) + 1
    i2 = i1 + size(A[i,1],1) - 1
    i1:i2
end

getblockrange(d, A::JotOpBlock, iblock) = reshape(d[blockrange(A,iblock)], range(A[i,1]))

function setblockrange!(d, A::JotOpBlock, iblock, dblock)
    d[blockrange(A,iblock)] .= dblock[:]
end

function blockdomain(A::JotOpBlock, iblock)
    i1 = iblock == 1 ? 1 : mapreduce(i->size(A[1,i],2), +, 1(iblock-1)) + 1
    i2 = i1 + size(A[1,iblock],1) - 1
    i1:i2
end

getblockdomain(m, A::JotOpBlock, iblock) = reshape(m[blockdomain(A,iblock)], domain(A[1,i]))

function setblockdomain!(m, A::JotOpBlock, iblock, mblock)
    m[blockdomain(A,iblock)] .= mblock[:]
end

export Jet, JotOp, JotOpLn, JotOpNl, JotSpace, JotOpZeroBlock, domain,
    getblockdomain, getblockrange, jacobian, jet, nblocks, setblockdomain!,
    setblockrange!, shape, state, state!

#
# test operators
#

# linear operator
function JotOpDiagonal(d)
    df!(d,m;kwargs...) = d .= kwargs[:diagonal] .* m
    spc = JotSpace(eltype(d),size(d))
    JotOpLn(;df! = df!, df′! = df!, dom = spc, rng = spc, s = (diagonal=d,))
end

# non-linear operator
function JotOpSquare(n)
    f!(d,m;kwargs...) = d .= m.^2
    df!(δd,δm;kwargs...) = δd .= 2 .* kwargs[:mₒ] .* δm
    spc = JotSpace(Float64, n)
    JotOpNl(;f! = f!, df! = df!, df′! = df!, dom = spc, rng = spc)
end

export JotOpDiagonal, JotOpSquare

end
