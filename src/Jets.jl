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
 * kron, +, -, show
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
Base.Array(R::JetAbstractSpace{T,N}) where {T,N} = Array{T,N}(undef, size(R))

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
    if isa(f!, typeof(jet_missing)) && isa(df!, typeof(jet_missing))
        error("must set at-least one of f! and df!")
    end
    if isa(f!, typeof(jet_missing))
        f! = df!
    end
    if isa(df′!, typeof(jet_missing))
        df′! = df!
    end
    Jet(dom, rng, f!, df!, df′!, Array{eltype(dom)}(undef, ntuple(i->0,ndims(dom))), s)
end

f!(d, jet::Jet, m; kwargs...) = jet.f!(d, m; kwargs...)
df!(d, jet::Jet, m; kwargs...) = jet.df!(d, m; kwargs...)
df′!(m, jet::Jet, d; kwargs...) = jet.df′!(m, d; kwargs...)

abstract type Jop{T<:Jet} end

struct JopNl{T<:Jet} <: Jop{T}
    jet::T
end
JopNl(;kwargs...) = JopNl(Jet(;kwargs...))

struct JopLn{T<:Jet} <: Jop{T}
    jet::T
end
JopLn(jet::Jet, mₒ::AbstractArray) = begin point!(jet, mₒ); JopLn(jet) end
JopLn(;kwargs...) = JopLn(Jet(;kwargs...))

struct JopAdjoint{J<:Jet,T<:Jop{J}} <: Jop{J}
    op::T
end

domain(jet::Jet) = jet.dom
Base.range(jet::Jet) = jet.rng
Base.eltype(jet::Jet) = promote_type(eltype(domain(jet)), eltype(range(jet)))
state(jet::Jet) = jet.s
state!(jet, s) = jet.s = merge(jet.s, s)
point(jet::Jet) = jet.mₒ
point!(jet::Jet, mₒ::AbstractArray) = jet.mₒ = mₒ

jet(A::Jop) = A.jet
jet(A::JopAdjoint) = jet(A.op)
Base.eltype(A::Jop) = eltype(jet(A))
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

jacobian(jet::Jet, mₒ::AbstractArray) = JopLn(jet, mₒ)
jacobian(F::JopNl, mₒ::AbstractArray) = jacobian(jet(F), mₒ)
jacobian(A::JopLn, mₒ::AbstractArray) = A

Base.adjoint(A::JopLn) = JopAdjoint(A)
Base.adjoint(A::JopAdjoint) = A.op

LinearAlgebra.mul!(d::AbstractArray, F::JopNl, m::AbstractArray) = f!(d, jet(F), m; state(F)...)
LinearAlgebra.mul!(d::AbstractArray, A::JopLn, m::AbstractArray) = df!(d, jet(A), m; mₒ=point(A), state(A)...)
LinearAlgebra.mul!(m::AbstractArray, A::JopAdjoint{J,T}, d::AbstractArray) where {J<:Jet,T<:JopLn} = df′!(m, jet(A), d; mₒ=point(A), state(A)...)

Base.:*(A::Jop, m::AbstractArray) = mul!(zeros(range(A)), A, m)

#
# composition
#
JetComposite(jets) = Jet(f! = JetComposite_f!, df! = JetComposite_df!, df′! = JetComposite_df′!, dom = domain(jets[end]), rng = range(jets[1]), s = (jets=jets,))

function JetComposite_f!(d::T, m; jets, kwargs...) where {T<:AbstractArray}
    g = mapreduce(i->(_m->f!(zeros(range(jets[i])), jets[i], _m; state(jets[i])...)), ∘, 1:length(jets))
    d .= g(m)
    d::T
end

function JetComposite_df!(d::T, m; jets, kwargs...) where {T<:AbstractArray}
    dg = mapreduce(i->(_m->df!(zeros(range(jets[i])), jets[i], _m; mₒ = point(jets[i]), state(jets[i])...)), ∘, 1:length(jets))
    d .= dg(m)
    d::T
end

function JetComposite_df′!(m::T, d; jets, kwargs...) where {T<:AbstractArray}
    dg′ = mapreduce(i->(_d->df′!(zeros(domain(jets[i])), jets[i], _d; mₒ = point(jets[i]), state(jets[i])...)), ∘, length(jets):-1:1)
    m .= dg′(d)
    m::T
end

jets(jet::Jet) = (jet,)
jets(jet::Jet{D,R,typeof(JetComposite_f!)}) where {D,R} = state(jet).jets
Base.:∘(jet₂::Jet, jet₁::Jet) = JetComposite((jets(jet₂)..., jets(jet₁)...))
Base.:∘(A₂::Jop, A₁::Jop) = JopNl(jet(A₂) ∘ jet(A₁))
Base.:∘(A₂::JopLn, A₁::JopLn) = JopLn(jet(A₂) ∘ jet(A₁))

function point!(jet::Jet{D,R,typeof(JetComposite_f!)}, mₒ::AbstractArray) where {D<:JetAbstractSpace,R<:JetAbstractSpace}
    jet.mₒ = mₒ
    jets = state(jet).jets
    _m = copy(mₒ)
    for i = 1:length(jets)
        point!(jets[i], _m)
        if i < length(jets)
            _m = f!(zeros(range(jets[i])), jets[i], _m; mₒ = point(jets[i]), state(jets[i])...)
        end
    end
    mₒ
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
space(R::JetBSpace, iblock::Integer) = R.spaces[iblock]

getblock!(x::AbstractArray, R::JetBSpace, iblock::Integer, xblock::AbstractArray) = xblock .= x[indices(R, iblock)]
getblock(x::AbstractArray, R::JetBSpace, iblock::Integer) = getblock!(x, R, iblock, Array(R.spaces[iblock]))
setblock!(x::AbstractArray, R::JetBSpace, iblock::Integer, xblock::AbstractArray) = x[indices(R, iblock)] .= xblock

function JetBlock(jets::AbstractMatrix{T}) where {T<:Jet}
    dom = JetBSpace([domain(jets[1,i]) for i=1:size(jets,2)])
    rng = JetBSpace([range(jets[i,1]) for i=1:size(jets,1)])
    Jet(f! = JetBlock_f!, df! = JetBlock_df!, df′! = JetBlock_df′!, dom = dom, rng = rng, s = (jets=jets,dom=dom,rng=rng))
end
JetBlock(jets::AbstractVector{T}) where {T<:Jet} = JetBlock(reshape(jets, length(jets), 1))
JopBlock(ops::AbstractArray{T}) where {T<:JopLn} = JopLn(JetBlock(jet.(ops)))
JopBlock(ops::AbstractArray{T}) where {T<:Jop} = JopNl(JetBlock(jet.(ops)))

JopZeroBlock(dom::JetSpace, rng::JetSpace) = JopLn(df! = JopZeroBlock_df!, dom = dom, rng = rng)
JopZeroBlock_df!(d, m; kwargs...) = d .= 0

Base.iszero(jet::Jet{D,R,typeof(JopZeroBlock_df!)}) where {D<:JetAbstractSpace,R<:JetAbstractSpace} = true
Base.iszero(jet::Jet) = false
Base.iszero(A::Jop) = iszero(jet(A))

macro blockop(ex)
    :(JopBlock($(esc(ex))))
end

function JetBlock_f!(d::AbstractArray, m::AbstractArray; jets, dom, rng, kwargs...)
    local dtmp
    if size(jets, 2) > 1
        dtmp = zeros(range(jets[1,1]))
    end
    for iblkrow = 1:size(jets, 1)
        _d = reshape(@view(d[indices(rng, iblkrow)]), range(jets[iblkrow,1]))
        if size(jets, 2) > 1
            dtmp = length(dtmp) == length(range(jets[iblkrow,1])) ? reshape(dtmp, range(jets[iblkrow,1])) : zeros(range(jets[iblkrow,1]))
        end
        for iblkcol = 1:size(jets, 2)
            if !iszero(jets[iblkrow,iblkcol])
                if size(jets, 2) == 1
                    f!(_d, jets[iblkrow,iblkcol], m; state(jets[iblkrow,iblkcol])...)
                else
                    _m = reshape(@view(m[indices(dom, iblkcol)]), domain(jets[iblkrow,iblkcol]))
                    _d .+= f!(dtmp, jets[iblkrow,iblkcol], _m; state(jets[iblkrow,iblkcol])...)
                end
            end
        end
    end
    d
end

function JetBlock_df!(d::AbstractArray, m::AbstractArray; jets, dom, rng, kwargs...)
    local dtmp
    if size(jets, 2) > 1
        dtmp = zeros(range(jets[1,1]))
    end
    for iblkrow = 1:size(jets, 1)
        _d = reshape(@view(d[indices(rng, iblkrow)]), range(jets[iblkrow,1]))
        if size(jets, 2) > 1
            dtmp = length(dtmp) == length(range(jets[iblkrow,1])) ? reshape(dtmp, range(jets[iblkrow,1])) : zeros(range(jets[iblkrow,1]))
        end
        for iblkcol = 1:size(jets, 2)
            if !iszero(jets[iblkrow,iblkcol])
                if size(jets, 2) == 1
                    df!(_d, jets[iblkrow,iblkcol], m; mₒ = point(jets[iblkrow,iblkcol]), state(jets[iblkrow,iblkcol])...)
                else
                    _m = reshape(@view(m[indices(dom, iblkcol)]), domain(jets[iblkrow,iblkcol]))
                    _d .+= df!(dtmp, jets[iblkrow,iblkcol], _m; mₒ = point(jets[iblkrow,iblkcol]), state(jets[iblkrow,iblkcol])...)
                end
            end
        end
    end
    d
end

function JetBlock_df′!(m::AbstractArray, d::AbstractArray; jets, dom, rng, kwargs...)
    local mtmp
    if size(jets, 1) > 1
        mtmp = zeros(domain(jets[1,1]))
    end
    for iblkcol = 1:size(jets, 2)
        _m = reshape(@view(m[indices(dom, iblkcol)]), domain(jets[1,iblkcol]))
        if size(jets, 1) > 1
            mtmp = length(mtmp) == length(domain(jets[1,iblkcol])) ? reshape(mtmp, domain(jets[1,iblkcol])) : zeros(domain(jets[1,iblkcol]))
        end
        for iblkrow = 1:size(jets, 1)
            if !iszero(jets[iblkrow,iblkcol])
                if size(jets, 1) == 1
                    df′!(_m, jets[iblkrow,iblkcol], d; mₒ=point(jets[iblkrow,iblkcol]), state(jets[iblkrow,iblkcol])...)
                else
                    _d = reshape(@view(d[indices(rng, iblkrow)]), range(jets[iblkrow,iblkcol]))
                    _m .+= df′!(mtmp, jets[iblkrow,iblkcol], _d; mₒ=point(jets[iblkrow,iblkcol]), state(jets[iblkrow,iblkcol])...)
                end
            end
        end
    end
    m
end

function point!(jet::Jet{D,R,typeof(JetBlock_f!)}, mₒ::AbstractArray) where {D<:JetAbstractSpace,R<:JetAbstractSpace}
    jets = state(jet).jets
    dom = domain(jet)
    for icol = 1:size(jets, 2), irow = 1:size(jets, 1)
        point!(jets[irow,icol], reshape(@view(mₒ[indices(dom, icol)]), domain(jets[irow,icol])))
    end
    mₒ
end

nblocks(jet::Jet{D,R,typeof(JetBlock_f!)}) where {D<:JetAbstractSpace,R<:JetAbstractSpace}= size(state(jet).jets)
nblocks(jet::Jet{D,R,typeof(JetBlock_f!)}, i) where {D<:JetAbstractSpace,R<:JetAbstractSpace} = size(state(jet).jets, i)
nblocks(A::Jop{T}) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetBlock_f!)}} = nblocks(jet(A))
nblocks(A::Jop{T}, i) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetBlock_f!)}} = nblocks(jet(A), i)

getblock(jet::Jet{D,R,typeof(JetBlock_f!)}, i, j) where {D,R} = state(jet).jets[i,j]
getblock(A::JopLn{T}, i, j) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetBlock_f!)}} = JopLn(getblock(jet(A), i, j))
getblock(::Type{JopNl}, A::Jop{T}, i, j) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetBlock_f!)}} = JopNl(getblock(jet(A), i, j))
getblock(::Type{JopLn}, A::Jop{T}, i, j) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetBlock_f!)}} = JopLn(getblock(jet(A), i, j))
getblock(A::JopAdjoint{Jet{D,R,typeof(JetBlock_f!)}}, i, j) where {D,R} = JopAdjoint(getindex(A.op, j, i))

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
        B[:,icol] .= mul!(d, A, m)
    end
    B
end

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

function linearity_test(A::JopLn)
    m1 = -1 .+ 2 * rand(domain(A))
    m2 = -1 .+ 2 * rand(range(A))
    lhs = A*(m1 + m2)
    rhs = A*m1 + A*m2
    lhs, rhs
end

export Jet, JetAbstractSpace, JetSpace, Jop, JopLn, JopNl, JopZeroBlock,
    @blockop, domain, getblock, getblock!, dot_product_test, getblock,
    getblock!, indices, jacobian, jet, linearity_test, nblocks, point,
    setblock!, shape, space, state, state!

end
