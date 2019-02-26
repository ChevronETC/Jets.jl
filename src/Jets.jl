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

struct JetSpace{T,N} <: JetAbstractSpace{T,N}
    n::NTuple{N,Int}
end
JetSpace(_T::Type{T}, n::Vararg{Int,N}) where {T,N} = JetSpace{T,N}(n)
JetSpace(_T::Type{T}, n::NTuple{N,Int}) where {T,N} = JetSpace{T,N}(n)

Base.size(R::JetSpace) = R.n
Base.eltype(R::Type{JetSpace{T,N}}) where {T,N} = T
Base.eltype(R::Type{JetSpace{T}}) where {T} = T

for f in (:ones, :rand, :zeros)
    @eval (Base.$f)(R::JetSpace{T,N}) where {T,N} = ($f)(T,size(R))::Array{T,N}
end
Base.Array(R::JetSpace{T,N}) where {T,N} = Array{T,N}(undef, size(R))

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
JopLn(jet::Jet, mₒ::AbstractArray) = JopLn(point!(jet, mₒ))
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
point!(jet::Jet, mₒ::AbstractArray) = begin jet.mₒ = mₒ; jet end

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

shape(A::AbstractMatrix,i) = (size(A, i),)
shape(A::AbstractMatrix) = ((size(A, 1),), (size(A, 2),))

Base.size(A::Union{Jet,Jop}, i) = prod(shape(A, i))
Base.size(A::Union{Jet,Jop}) = (size(A, 1), size(A, 2))

jacobian(jet::Jet, mₒ::AbstractArray) = JopLn(jet, mₒ)
jacobian(F::JopNl, mₒ::AbstractArray) = jacobian(jet(F), mₒ)
jacobian(A::Union{JopLn,AbstractMatrix}, mₒ::AbstractArray) = A

Base.adjoint(A::JopLn) = JopAdjoint(A)
Base.adjoint(A::JopAdjoint) = A.op

LinearAlgebra.mul!(d::AbstractArray, F::JopNl, m::AbstractArray) = f!(d, jet(F), m; state(F)...)
LinearAlgebra.mul!(d::AbstractArray, A::JopLn, m::AbstractArray) = df!(d, jet(A), m; mₒ=point(A), state(A)...)
LinearAlgebra.mul!(m::AbstractArray, A::JopAdjoint{J,T}, d::AbstractArray) where {J<:Jet,T<:JopLn} = df′!(m, jet(A), d; mₒ=point(A), state(A)...)

Base.:*(A::Jop, m::AbstractArray) = mul!(zeros(range(A)), A, m)

#
# Symmetric spaces / arrays
#
struct JetSSpace{T,N,F<:Function} <: JetAbstractSpace{T,N}
    n::NTuple{N,Int}
    M::NTuple{N,Int}
    map::F
end
JetSSpace(_T::Type{T}, n::NTuple{N,Int}, M::NTuple{N,Int}, map::F) where {T,N,F} = JetSSpace{T,N,F}(n, M, map)

Base.size(R::JetSSpace) = R.n
Base.eltype(R::Type{JetSSpace{T,N,F}}) where {T,N,F} = T
Base.eltype(R::Type{JetSSpace{T,N}}) where {T,N} = T
Base.eltype(R::Type{JetSSpace{T}}) where {T} = T

struct SymmetricArray{T,N,F<:Function} <: AbstractArray{T,N}
    A::Array{T,N}
    n::NTuple{N,Int}
    map::F
end

# SymmetricArray array interface implementation <--
Base.IndexStyle(::Type{T}) where {T<:SymmetricArray} = IndexCartesian()
Base.size(A::SymmetricArray) = A.n
Base.getindex(A::SymmetricArray{T,N}, I::Vararg{Int,N}) where {T,N} = A.A[A.map(I)]
Base.setindex!(A::SymmetricArray{T,N}, v, I::Vararg{Int,N}) where {T,N} = A.A[A.map(I)] = v
# -->

# SymmetricArray broadcasting interface implementation --<
Base.BroadcastStyle(::Type{<:SymmetricArray}) = Broadcast.ArrayStyle{SymmetricArray}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{SymmetricArray}}, ::Type{T}) where {T}
    A = find_symmetricarray(bc)
    SymmetricArray(similar(Array{T}, axes(bc)), A.n, A.map)
end
find_symmetricarray(bc::Broadcast.Broadcasted) = find_symmetricarray(bc.args)
find_symmetricarray(args::Tuple) = find_symmetricarray(find_symmetricarray(args[1]), Base.tail(args))
find_symmetricarray(x) = x
find_symmetricarray(a::SymmetricArray, rest) = a
find_symmetricarray(::Any, rest) = find_symmetricarray(rest)

Base.similar(A::SymmetricArray) = SymmetricArray(similar(A.A), A.n, A.map)
# -->

for f in (:ones, :rand, :zeros)
    @eval (Base.$f)(R::JetSSpace{T,N,F}) where {T,N,F} = SymmetricArray(($f)(T,R.M), R.n, R.map)::SymmetricArray{T,N,F}
end
Base.Array(R::JetSSpace{T,N,F}) where {T,N,F} = SymmetricArray{T,N,F}(Array{T,N}(undef, R.M), R.n, R.map)

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
    jet
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

indices(R::JetBSpace, iblock::Integer) = R.indices[iblock]
space(R::JetBSpace, iblock::Integer) = R.spaces[iblock]
nblocks(R::JetBSpace) = length(R.spaces)

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

Base.similar(A::BlockArray) = BlockArray([similar(A.arrays[i]) for i=1:length(A.arrays)], A.indices)

LinearAlgebra.norm(x::BlockArray, p::Real=2) = mapreduce(_x->norm(_x,p)^p, +, x.arrays)^(1/p)

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
# -->

# BlockArray broadcasting implementation --<
struct BlockArrayStyle <: Broadcast.AbstractArrayStyle{1} end
Base.BroadcastStyle(::Type{<:BlockArray}) = BlockArrayStyle()
BlockArrayStyle(::Val{1}) = BlockArrayStyle()

Base.similar(bc::Broadcast.Broadcasted{BlockArrayStyle}, ::Type{T}) where {T} = similar(find_blockarray(bc)::BlockArray{T})
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
        _d = getblock(d, iblkrow)
        if size(jets, 2) > 1
            dtmp = length(dtmp) == length(range(jets[iblkrow,1])) ? reshape(dtmp, range(jets[iblkrow,1])) : zeros(range(jets[iblkrow,1]))
        end
        for iblkcol = 1:size(jets, 2)
            _m = getblock(m, iblkcol)
            if size(jets, 2) > 1
                _d .+= f!(dtmp, jets[iblkrow,iblkcol], _m; state(jets[iblkrow,iblkcol])...)
            else
                f!(_d, jets[iblkrow,iblkcol], m; state(jets[iblkrow,iblkcol])...)
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
        _d = getblock(d, iblkrow)
        if size(jets, 2) > 1
            dtmp = length(dtmp) == length(range(jets[iblkrow,1])) ? reshape(dtmp, range(jets[iblkrow,1])) : zeros(range(jets[iblkrow,1]))
        end
        for iblkcol = 1:size(jets, 2)
            _m = getblock(m, iblkcol)
            if !iszero(jets[iblkrow,iblkcol])
                if size(jets, 2) > 1
                    _d .+= df!(dtmp, jets[iblkrow,iblkcol], _m; mₒ = point(jets[iblkrow,iblkcol]), state(jets[iblkrow,iblkcol])...)
                else
                    df!(_d, jets[iblkrow,iblkcol], _m; mₒ = point(jets[iblkrow,iblkcol]), state(jets[iblkrow,iblkcol])...)
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
        _m = getblock(m, iblkcol)
        if size(jets, 1) > 1
            mtmp = length(mtmp) == length(domain(jets[1,iblkcol])) ? reshape(mtmp, domain(jets[1,iblkcol])) : zeros(domain(jets[1,iblkcol]))
        end
        for iblkrow = 1:size(jets, 1)
            _d = getblock(d, iblkrow)
            if !iszero(jets[iblkrow,iblkcol])
                if size(jets, 1) > 1
                    _m .+= df′!(mtmp, jets[iblkrow,iblkcol], _d; mₒ=point(jets[iblkrow,iblkcol]), state(jets[iblkrow,iblkcol])...)
                else
                    df′!(_m, jets[iblkrow,iblkcol], _d; mₒ=point(jets[iblkrow,iblkcol]), state(jets[iblkrow,iblkcol])...)
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
        point!(jets[irow,icol], getblock(mₒ, icol))
    end
    jet
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
