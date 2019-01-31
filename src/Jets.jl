#=
NOTES...
* we use an extra field (a named tuple) in JotOp to keep operator state
* I could not figure out how to make the block operator without including spaces in Jop
* Can we use LinearMaps.jl, AbstractOperators.jl, etc.?

TODO...
 * revisit type stability / generated code
   - compoiste operator has unstable mul,mul!
   - block operator has unstable constructor
 * LinearAlgebra has an Adjoint type, can we use it?
 * propagating @inbounds
 * kron, +, -
=#

module Jets

using LinearAlgebra

abstract type JetAbstractSpace{T,N} end

Base.eltype(R::JetAbstractSpace{T}) where {T} = T
Base.eltype(R::Type{JetAbstractSpace{T,N}}) where {T,N} = T
Base.eltype(R::Type{JetAbstractSpace{T}}) where {T} = T
Base.ndims(R::JetAbstractSpace{T,N}) where {T,N} = N
Base.length(R::JetAbstractSpace) = prod(size(R))
Base.reshape(x::AbstractArray, R::JetAbstractSpace) = reshape(x, size(R))

for f in (:ones, :rand, :zeros)
    @eval (Base.$f)(R::JetAbstractSpace{T,N}) where {T,N} = ($f)(T,size(R))::Array{T,N}
end

struct JetSpace{T,N} <: JetAbstractSpace{T,N}
    n::NTuple{N,Int}
end
JetSpace(_T::Type{T}, n::Vararg{Int,N}) where {T,N} = JetSpace{T,N}(n)
JetSpace(_T::Type{T}, n::NTuple{N,Int}) where {T,N} = JetSpace{T,N}(n)

Base.size(R::JetSpace) = R.n
Base.eltype(R::Type{JetSpace{T,N}}) where {T,N} = T
Base.eltype(R::Type{JetSpace{T}}) where {T} = T

jet_missing(m) = error("not implemented")

mutable struct Jet{D<:JetAbstractSpace,R<:JetAbstractSpace,F<:Function,DF<:Function,DF′<:Function,M<:AbstractArray,S<:NamedTuple}
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
struct JopComposite{T<:Tuple, D<:JetAbstractSpace, R<:JetAbstractSpace} <: Jop
    ops::T
    function JopComposite(ops::T) where {T<:Tuple}
        D = typeof(domain(ops[end]))
        R = typeof(range(ops[1]))
        new{T,D,R}(ops)
    end
end
operators(F::Jop) = (F,)
operators(F::JopComposite) = F.ops
Base.:∘(A₂::Jop, A₁::Jop) = JopComposite((operators(A₂)..., operators(A₁)...))

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
Base.eltype(R::Type{JetBSpace{T,N}}) where {T,N} = T
Base.eltype(R::Type{JetBSpace{T}}) where {T} = T

indices(R::JetBSpace, iblock::Integer) = R.indices[iblock]

block(x::AbstractArray, R::JetBSpace, iblock::Integer) = reshape(x[indices(R, iblock)], R.spaces[iblock])
block!(x::AbstractArray, R::JetBSpace, iblock::Integer, xblock::AbstractArray) = x[indices(R, iblock)] .= xblock

struct JopBlock{D<:JetBSpace,R<:JetBSpace,T<:Jop} <: Jop
    dom::D
    rng::R
    ops::Matrix{T}
end
function JopBlock(ops::Matrix{T}) where {T<:Jop}
    dom = JetBSpace([domain(ops[1,i]) for i=1:size(ops,2)])
    rng = JetBSpace([range(ops[i,1]) for i=1:size(ops,1)])
    JopBlock(dom, rng, ops)
end
JopBlock(ops::Vector{T}) where {T<:Jop} = JopBlock(reshape(ops, length(ops), 1))

struct JopZeroBlock{T,N} <: Jop
    dom::JetSpace{T,N}
    rng::JetSpace{T,N}
end
jacobian(A::JopZeroBlock, m) = A

domain(A::JopZeroBlock) = A.dom
Base.range(A::JopZeroBlock) = A.rng

macro blockop(ex)
    :(JopBlock($(esc(ex))))
end

Base.getindex(A::JopBlock, i, j) = A.ops[i,j]
Base.getindex(A::JopAdjoint{T}, i, j) where {T<:JopBlock} = A.op.ops[j,i]

domain(A::JopBlock) = A.dom
Base.range(A::JopBlock) = A.rng

nblocks(A::JopBlock) = size(A.ops)
nblocks(A::JopBlock, i) = size(A.ops, i)

function LinearAlgebra.mul!(d::AbstractArray, A::JopBlock, m::AbstractArray)
    dom,rng = domain(A),range(A)
    dtmp = nblocks(A, 1) == 1 ? d : zeros(range(A[1,1]))
    for iblkrow = 1:nblocks(A, 1)
        _d = reshape(@view(d[indices(rng, iblkrow)]), range(A[iblkrow,1]))
        dtmp = length(dtmp) == length(range(A[iblkrow,1])) ? reshape(dtmp, range(A[iblkrow,1])) : zeros(range(A[iblkrow,1]))
        for iblkcol = 1:nblocks(A, 2)
            if !isa(A[iblkrow,iblkcol], JopZeroBlock)
                if nblocks(A, 2) == 1
                    mul!(_d, A[iblkrow,iblkcol], m)
                else
                    _m = reshape(@view(m[indices(dom, iblkcol)]), domain(A[iblkrow,iblkcol]))
                    _d .+= mul!(dtmp, A[iblkrow,iblkcol], _m)
                end
            end
        end
    end
    d
end

function LinearAlgebra.mul!(m::AbstractArray, B::JopAdjoint{T}, d::AbstractArray) where {T<:JopBlock}
    A = B.op
    dom,rng = domain(A),range(A)
    mtmp = nblocks(A, 1) == 1 ? m : zeros(domain(A[1,1]))
    for iblkcol = 1:nblocks(A, 2)
        _m = reshape(@view(m[indices(dom, iblkcol)]), domain(A[1,iblkcol]))
        mtmp = length(mtmp) == length(domain(A[1,iblkcol])) ? reshape(mtmp, domain(A[1,iblkcol])) : zeros(domain(A[1,iblkcol]))
        for iblkrow = 1:nblocks(A, 1)
            if !isa(A[iblkrow,iblkcol], JopZeroBlock)
                if nblocks(A, 1) == 1
                    mul!(_m, A[iblkrow,iblkcol], d)
                else
                    _d = reshape(@view(d[indices(rng, iblkrow)]), range(A[iblkrow,iblkcol]))
                    _m .+= mul!(mtmp, adjoint(A[iblkrow,iblkcol]), _d)
                end
            end
        end
    end
    m
end

Base.adjoint(A::JopBlock{D,R,T}) where {D,R,T} = JopAdjoint(A)

function jacobian(F::JopBlock, m::AbstractArray)
    dom = domain(F)
    JopBlock([jacobian(F[i,j], reshape(@view(m[indices(dom, j)]), domain(F[i,j]))) for i=1:nblocks(F,1), j=1:nblocks(F,2)])
end

#
# utilities
#
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

export Jet, JetSpace, Jop, JopLn, JopNl, JopZeroBlock, @blockop, domain, block,
    block!, dot_product_test, jacobian, jet, nblocks, point, setblockdomain!,
    setblockrange!, shape, state, state!

end
