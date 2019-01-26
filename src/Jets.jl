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

module Jets

using LinearAlgebra

struct JetSpace{T,N}
    n::NTuple{N,Int}
end
JetSpace(_T::Type{T}, n::Vararg{Int,N}) where {T,N} = JetSpace{T,N}(n)
JetSpace(_T::Type{T}, n::NTuple{N,Int}) where {T,N} = JetSpace{T,N}(n)

Base.size(R::JetSpace) = R.n
Base.eltype(R::JetSpace{T}) where {T} = T
Base.eltype(R::Type{JetSpace{T,N}}) where {T,N} = T
Base.eltype(R::Type{JetSpace{T}}) where {T} = T
Base.ndims(R::JetSpace{T,N}) where {T,N} = N
Base.reshape(x::AbstractArray, R::JetSpace) = reshape(x, size(R))

for f in (:ones, :rand, :zeros)
    @eval (Base.$f)(R::JetSpace{T}) where {T} = ($f)(T,R.n)
end

jet_missing(m) = error("not implemented")

mutable struct Jet{D<:JetSpace,R<:JetSpace,F<:Function,DF<:Function,DF′<:Function,M<:AbstractArray,S<:NamedTuple}
    dom::D
    rng::R
    f!::F
    df!::DF
    df′!::DF′
    mₒ::M
    s::S
end
function Jet(;
        dom,
        rng,
        f! = jet_missing,
        df! = jet_missing,
        df′! = jet_missing,
        s = NamedTuple())
    Jet(dom, rng, f!, df!, df′!, Array{eltype(dom)}(undef, ntuple(i->0,ndims(dom))), s)
end

abstract type Jop end

struct JopNl{T<:Jet} <: Jop
    jet::T
end
JopNl(;kwargs...) = JopNl(Jet(;kwargs...))

struct JopLn{T<:Jet} <: Jop
    jet::T
end
JopLn(jet::Jet, mₒ::AbstractArray) = begin point!(jet, mₒ); JopLn(jet) end
JopLn(;kwargs...) = JopLn(Jet(;kwargs...))

struct JopAdjoint{T<:Jop} <: Jop
    op::T
end

domain(jet::Jet) = jet.dom
Base.range(jet::Jet) = jet.rng
state(jet::Jet) = jet.s
state!(jet, s) = jet.s = merge(jet.s, s)
point(jet::Jet) = jet.mₒ
point!(jet::Jet, mₒ::AbstractArray) = jet.mₒ = mₒ

jet(A::Jop) = A.jet
jet(A::JopAdjoint) = jet(A.op)
point(A::JopLn) = point(jet(A))
point(A::JopAdjoint) = point(jet(A.op))
state(A::Jop) = state(jet(A))
state!(A::Jop, s) = state!(jet(A), s)

domain(A::Jop) = domain(jet(A))
Base.range(A::Jop) = range(jet(A))

domain(A::JopAdjoint) = range(A.op)
Base.range(A::JopAdjoint) = domain(A.op)

function shape(A::Union{Jet,Jop}, i)
    if i == 1
        return size(range(A))
    end
    size(domain(A))
end
shape(A::Union{Jet,Jop}) = (shape(A, 1), shape(A, 2))

shape(A::AbstractMatrix,i) = size(A, i)
shape(A::AbstractMatrix) = size(A)

Base.size(A::Union{Jet,Jop}, i) = prod(shape(A, i))
Base.size(A::Union{Jet,Jop}) = (size(A, 1), size(A, 2))

jacobian(F::JopNl, mₒ::AbstractArray) = JopLn(jet(F), mₒ)
jacobian(A::JopLn, mₒ::AbstractArray) = A

Base.adjoint(A::JopLn) = JopAdjoint(A)
Base.adjoint(A::JopAdjoint) = A.op

LinearAlgebra.mul!(d::AbstractArray, F::JopNl, m::AbstractArray) = jet(F).f!(d, m; state(F)...)
LinearAlgebra.mul!(d::AbstractArray, A::JopLn, m::AbstractArray) = jet(A).df!(d, m; mₒ=point(A), state(A)...)
LinearAlgebra.mul!(m::AbstractArray, A::JopAdjoint{T}, d::AbstractArray) where {T<:JopLn} = jet(A).df′!(m, d; mₒ=point(A), state(A)...)

Base.:*(A::Jop, m::AbstractArray) = mul!(zeros(range(A)), A, m)

#
# composition
#
struct JopComposite{T<:Tuple, D<:JetSpace, R<:JetSpace} <: Jop
    ops::T
    function JopComposite(ops::T) where {T<:Tuple}
        D = typeof(domain(ops[end]))
        R = typeof(range(ops[1]))
        new{T,D,R}(ops)
    end
end
_operators(F::Jop) = (F,)
_operators(F::JopComposite) = F.ops
Base.:∘(A₂::Jop, A₁::Jop) = JopComposite((_operators(A₂)..., _operators(A₁)...))

domain(A::JopComposite) = domain(A.ops[end])
Base.range(A::JopComposite) = range(A.ops[1])

function LinearAlgebra.mul!(d::T, A::JopComposite, m::AbstractArray) where {T<:AbstractArray}
    f = mapreduce(i->(_m->A.ops[i]*_m), ∘, 1:length(A.ops))
    d .= f(m)
    d::T
end

function LinearAlgebra.mul!(m::T, A::JopAdjoint{O}, d::AbstractArray) where {T<:AbstractArray, O<:JopComposite}
    f = mapreduce(i->(_d->adjoint(A.op.ops[i])*_d), ∘, length(A.op.ops):-1:1)
    m .= f(d)
    m::T
end

Base.adjoint(A::JopComposite) = JopAdjoint(A)

function jacobian(A::JopComposite, m::AbstractArray)
    ops = []
    for i = 1:length(A.ops)
        push!(ops, jacobian(A.ops[i], m))
        if i < length(A.ops)
            m = A.ops[i]*m
        end
    end
    JopComposite((ops...,))
end

#
# Block operator
#
struct JopBlock{D<:JetSpace,R<:JetSpace,T<:Jop} <: Jop
    ops::Matrix{T}
    function JopBlock(ops::Matrix{T}) where {T<:Jop}
        D = promote_type(ntuple(i->eltype(range(ops[i,1])), size(ops,1))...)
        R = promote_type(ntuple(i->eltype(range(ops[1,i])), size(ops,2))...)
        new{JetSpace{D,1},JetSpace{R,1},T}(ops)
    end
end

struct JopZeroBlock{T,N} <: Jop
    dom::JetSpace{T,N}
    rng::JetSpace{T,N}
end
jacobian(A::JopZeroBlock, m) = A

macro blockop(ex)
    :(JopBlock($(esc(ex))))
end

Base.getindex(A::JopBlock, i, j) = A.ops[i,j]
Base.getindex(A::JopAdjoint{T}, i, j) where {T<:JopBlock} = A.op.ops[j,i]

function domain(A::JopBlock{D,R,T}) where {D,R,T}
    n = mapreduce(i->size(A[1,i],2), +, 1:nblocks(A,2))
    JetSpace(eltype(D),n)
end
function Base.range(A::JopBlock{D,R,T}) where {D,R,T}
    n = mapreduce(i->size(A[i,1],1), +, 1:nblocks(A,1))
    JetSpace(eltype(R),n)
end

nblocks(A::JopBlock) = size(A.ops)
nblocks(A::JopBlock, i) = size(A.ops, i)

function LinearAlgebra.mul!(d::AbstractArray, A::JopBlock, m::AbstractArray)
    rngN = 0
    for iblkrow = 1:nblocks(A,1)
        rng1 = rngN + 1
        rngN = rng1 + size(A[iblkrow,1],1) - 1
        _d = @view d[rng1:rngN]

        domN = 0
        for iblkcol = 1:nblocks(A,2)
            dom1 = domN + 1
            domN = dom1 + size(A[1,iblkcol],2) - 1
            if !isa(A[iblkrow,iblkcol], JopZeroBlock)
                _m = @view m[dom1:domN]
                _d .+= A[iblkrow,iblkcol] * reshape(_m, domain(A[iblkrow,iblkcol]))
            end
        end
    end
    d
end

function LinearAlgebra.mul!(m::AbstractArray, B::JopAdjoint{T}, d::AbstractArray) where {T<:JopBlock}
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
            if !isa(A[iblkrow,iblkcol], JopZeroBlock)
                _d = @view d[rng1:rngN]
                _m .+= adjoint(A[iblkrow,iblkcol]) * reshape(_d, range(A[iblkrow,iblkcol]))
            end
        end
    end
    m
end

Base.adjoint(A::JopBlock{D,R,T}) where {D,R,T} = JopAdjoint(A)

function jacobian(F::JopBlock, m::AbstractArray)
    domrng = Vector{UnitRange}(undef, nblocks(F, 2))
    domN = 0
    for i = 1:nblocks(F, 2)
        dom1 = domN + 1
        domN = dom1 + size(F[1,i], 2) - 1
        domrng[i] = dom1:domN
    end
    JopBlock([jacobian(F[i,j], @view(m[domrng[j]])) for i=1:nblocks(F,1), j=1:nblocks(F,2)])
end

function blockrange(A::JopBlock, iblock::Integer)
    i1 = (iblock == 1 ? 1 : mapreduce(i->size(A[i,1],1), +, 1:(iblock-1)) + 1)
    i2 = i1 + size(A[iblock,1],1) - 1
    i1:i2
end

getblockrange(d::AbstractArray, A::JopBlock, iblock::Integer) = reshape(d[blockrange(A,iblock)], range(A[iblock,1]))

function setblockrange!(d::AbstractArray, A::JopBlock, iblock::Integer, dblock::AbstractArray)
    d[blockrange(A,iblock)] .= dblock[:]
end

function blockdomain(A::JopBlock, iblock::Integer)
    i1 = iblock == 1 ? 1 : mapreduce(i->size(A[1,i],2), +, 1(iblock-1)) + 1
    i2 = i1 + size(A[1,iblock],1) - 1
    i1:i2
end

getblockdomain(m::AbstractArray, A::JopBlock, iblock::Integer) = reshape(m[blockdomain(A,iblock)], domain(A[1,iblock]))

function setblockdomain!(m::AbstractArray, A::JopBlock, iblock::Integer, mblock::AbstractArray)
    m[blockdomain(A,iblock)] .= mblock[:]
end

export Jet, JetSpace, Jop, JopLn, JopNl, JopZeroBlock, @blockop, domain,
    getblockdomain, getblockrange, jacobian, jet, nblocks, point,
    setblockdomain!, setblockrange!, shape, state, state!

end
