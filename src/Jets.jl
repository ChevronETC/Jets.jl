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

using CRC32c, LinearAlgebra

abstract type JetAbstractSpace{T,N} end

Base.eltype(R::JetAbstractSpace{T}) where {T} = T
Base.eltype(R::Type{JetAbstractSpace{T,N}}) where {T,N} = T
Base.eltype(R::Type{JetAbstractSpace{T}}) where {T} = T
Base.ndims(R::JetAbstractSpace{T,N}) where {T,N} = N
Base.length(R::JetAbstractSpace) = prod(size(R))
Base.size(R::JetAbstractSpace, i) = size(R)[i]
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
JopNl(;kwargs...) = JopNl(Jet(;kwargs...))

struct JopLn{T<:Jet} <: Jop{T}
    jet::T
end
JopLn(jet::Jet, mₒ::AbstractArray) = JopLn(point!(jet, mₒ))
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

domain(jet::Jet) = jet.dom
Base.range(jet::Jet) = jet.rng
Base.eltype(jet::Jet) = promote_type(eltype(domain(jet)), eltype(range(jet)))
state(jet::Jet) = jet.s
state!(jet, s) = begin jet.s = merge(jet.s, s); jet end
perfstat(jet::T) where {D,R,F<:Function,T<:Jet{D,R,F}} = Dict()
point(jet::Jet) = jet.mₒ
Base.close(jet::Jet) = finalize(jet)

function point!(jet::Jet, mₒ::AbstractArray)
    jet.mₒ = mₒ
    jet.upstate!(mₒ, state(jet))
    jet
end

jet(A::Jop) = A.jet
jet(A::JopAdjoint) = jet(A.op)
Base.eltype(A::Jop) = eltype(jet(A))
point(A::JopLn) = point(jet(A))
point(A::JopAdjoint) = point(jet(A.op))
state(A::Jop) = state(jet(A))
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
shape(A::Union{Jet,Jop}) = (shape(A, 1), shape(A, 2))

shape(A::AbstractMatrix,i) = (size(A, i),)
shape(A::AbstractMatrix) = ((size(A, 1),), (size(A, 2),))

Base.size(A::Union{Jet,Jop}, i) = prod(shape(A, i))
Base.size(A::Union{Jet,Jop}) = (size(A, 1), size(A, 2))

jacobian!(jet::Jet, mₒ::AbstractArray) = JopLn(jet, mₒ)
jacobian!(F::JopNl, mₒ::AbstractArray) = jacobian!(jet(F), mₒ)
jacobian!(A::Union{JopLn,AbstractMatrix}, mₒ::AbstractArray) = A

jacobian(F::Union{Jet,Jop}, mₒ::AbstractArray) = jacobian!(copy(F, false), copy(mₒ))
jacobian(A::AbstractMatrix, mₒ::AbstractArray) = copy(A)

Base.adjoint(A::JopLn) = JopAdjoint(A)
Base.adjoint(A::JopAdjoint) = A.op

LinearAlgebra.mul!(d::AbstractArray, F::JopNl, m::AbstractArray) = f!(d, jet(F), m; state(F)...)
LinearAlgebra.mul!(d::AbstractArray, A::JopLn, m::AbstractArray) = df!(d, jet(A), m; mₒ=point(A), state(A)...)
LinearAlgebra.mul!(m::AbstractArray, A::JopAdjoint{J,T}, d::AbstractArray) where {J<:Jet,T<:JopLn} = df′!(m, jet(A), d; mₒ=point(A), state(A)...)

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

Base.parent(A::SymmetricArray) = A.A

# SymmetricArray array interface implementation <--
Base.IndexStyle(::Type{T}) where {T<:SymmetricArray} = IndexCartesian()
Base.size(A::SymmetricArray) = A.n

function Base.getindex(A::SymmetricArray{T}, I::Vararg{Int}) where {T}
    for idim = 1:ndims(A)
        if I[idim] > size(A.A, idim)
            return conj(A.A[A.map(I)])
        end
    end
    A.A[CartesianIndex(I)]
end

function Base.setindex!(A::SymmetricArray{T}, v, I::Vararg{Int}) where {T}
    for idim = 1:ndims(A)
        if I[idim] > size(A.A, idim)
            A.A[A.map(I)] = conj(v)
            return A.A[A.map(I)]
        end
    end
    A.A[CartesianIndex(I)] = v
end
# -->

# SymmetricArray broadcasting interface implementation --<
Base.BroadcastStyle(::Type{<:SymmetricArray}) = Broadcast.ArrayStyle{SymmetricArray}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{SymmetricArray}}, ::Type{T}) where {T}
    A = find_symmetricarray(bc)
    SymmetricArray(similar(A.A), A.n, A.map)
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

jops(op::Jop{J}) where {D,R,J<:Jet{D,R,typeof(JetComposite_f!)}} = state(jet(op)).ops
jops(op::Jop) = (op,)
Base.:∘(A₂::Union{JopAdjoint,JopLn}, A₁::Union{JopAdjoint,JopLn}) = JopLn(JetComposite((jops(A₂)..., jops(A₁)...)))
Base.:∘(A₂::Jop, A₁::Jop) = JopNl(JetComposite((jops(A₂)..., jops(A₁)...)))
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

function JetBlock(ops::AbstractMatrix{T}; kwargs...) where {T<:Jop}
    dom = size(ops,2) == 1 ? domain(ops[1,1]) : JetBSpace([domain(ops[1,i]) for i=1:size(ops,2)])
    rng = JetBSpace([range(ops[i,1]) for i=1:size(ops,1)])
    Jet(f! = JetBlock_f!, df! = JetBlock_df!, df′! = JetBlock_df′!, dom = dom, rng = rng, s = (ops=ops,dom=dom,rng=rng))
end
JopBlock(ops::AbstractMatrix{T}; kwargs...) where {T<:Union{JopLn,JopAdjoint}} = JopLn(JetBlock(ops; kwargs...))
JopBlock(ops::AbstractMatrix{T}; kwargs...) where {T<:Jop} = JopNl(JetBlock(ops; kwargs...))
JopBlock(ops::AbstractVector{T}; kwargs...) where {T<:Jop} = JopBlock(reshape(ops, length(ops), 1); kwargs...)

JopZeroBlock(dom::JetSpace, rng::JetSpace) = JopLn(df! = JopZeroBlock_df!, dom = dom, rng = rng)
JopZeroBlock_df!(d, m; kwargs...) = d .= 0

Base.iszero(jet::Jet{D,R,typeof(JopZeroBlock_df!)}) where {D<:JetAbstractSpace,R<:JetAbstractSpace} = true
Base.iszero(jet::Jet) = false
Base.iszero(A::Jop) = iszero(jet(A))

macro blockop(ex)
    :(JopBlock($(esc(ex))))
end

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
            dtmp = length(dtmp) == length(range(ops[iblkrow,1])) ? reshape(dtmp, range(ops[iblkrow,1])) : zeros(range(ops[iblkrow,1]))
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
            dtmp = length(dtmp) == length(range(ops[iblkrow,1])) ? reshape(dtmp, range(ops[iblkrow,1])) : zeros(range(ops[iblkrow,1]))
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
            mtmp = length(mtmp) == length(domain(ops[1,iblkcol])) ? reshape(mtmp, domain(ops[1,iblkcol])) : zeros(domain(ops[1,iblkcol]))
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

nblocks(jet::Jet{D,R,typeof(JetBlock_f!)}) where {D<:JetAbstractSpace,R<:JetAbstractSpace}= size(state(jet).ops)
nblocks(jet::Jet{D,R,typeof(JetBlock_f!)}, i) where {D<:JetAbstractSpace,R<:JetAbstractSpace} = size(state(jet).ops, i)
nblocks(A::Jop{T}) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetBlock_f!)}} = nblocks(jet(A))
nblocks(A::Jop{T}, i) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetBlock_f!)}} = nblocks(jet(A), i)

getblock(jet::Jet{D,R,typeof(JetBlock_f!)}, i, j) where {D,R} = state(jet).ops[i,j]
getblock(A::JopLn{T}, i, j) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetBlock_f!)}} = getblock(jet(A), i, j)::JopLn
getblock(A::JopNl{T}, i, j) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetBlock_f!)}} = getblock(jet(A), i, j)
getblock(::Type{JopNl}, A::Jop{T}, i, j) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetBlock_f!)}} = getblock(jet(A), i, j)::JopNl
getblock(::Type{JopLn}, A::Jop{T}, i, j) where {T<:Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JetBlock_f!)}} = getblock(jet(A), i, j)::JopLn
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

function linearization_test(F::JopNl, mₒ::AbstractArray;
        μ=[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125], mmask=[], dmask = [], seed=Inf)
    mmask = length(mmask) == 0 ? ones(domain(F)) : mmask
    dmask = length(dmask) == 0 ? ones(range(F)) : dmask

    isfinite(seed) && Random.seed!(seed)
    δm = mmask .* (-1 .+ 2 .* rand(domain(F)))
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

export Jet, JetAbstractSpace, JetSpace, JetSSpace, Jop, JopAdjoint, JopLn, JopNl,
    JopZeroBlock, @blockop, domain, getblock, getblock!, dot_product_test, getblock,
    getblock!, indices, jacobian, jacobian!, jet, linearity_test, linearization_test,
    nblocks, perfstat, point, setblock!, shape, space, state, state!, symspace

end
