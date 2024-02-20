module RrulesExt


using Jets
using ChainRulesCore


"""
    Reverse differentiaiton rule for Jets.JopLn

This function is the core of AD in ChainRules and used by the Backend of Zygote.jl, Flux.jl
and other Julia AD systems. This is the recommended implementation for general AD support within
the Julia ecosystem.

This defines the rule so that a linear operator can be used in combination with networks and layers.
The operator itseld is not considered differentiable, only its action.

It defines the rule:
    (F*x)' . dy = F' dy


This allows the implementation of machine learning workflows such as:


```julia
using Jets, JetPackDSP, Flux, JetPackTransforms, JetPack, LinearAlgebra

A = JopDiagonal(rand(Float32, 64))
# create a random vector in the domain of operator A
m = rand(domain(A))
m0 = rand(domain(A))

# apply the forward lineare map of operator A to domain vector m, returning range vector d
d = A*m

# Gradient (Flux returns a tuple)
g = gradient(x -> mse(A*x - d), m0)[1]

diff = norm(g - A'*(A*m0 - d))
println("Difference between true and Flux gradient: ", diff)

```

"""
function ChainRulesCore.rrule(::typeof(*), F::T, x) where {T<:Jets.JopLn}
    y = F*x
    bck(Δy) = (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), F'*Δy)
    return y, bck
end


"""
    Reverse differentiaiton rule for Jets.JopNln

This function is the core of AD in ChainRules and used by the Backend of Zygote.jl, Flux.jl
and other Julia AD systems. This is the recommended implementation for general AD support within
the Julia ecosystem.

This defines the rule so that a linear operator can be used in combination with networks and layers.
The operator itseld is not considered differentiable, only its action.

It defines the rule:
    (F(x))' . dy = jaconian(F, x)' dy


This allows the implementation of machine learning workflows such as:


```julia
using Jets, JetPackDSP, Flux, JetPackTransforms, JetPack, LinearAlgebra

F = JopEnvelope(JetSpace(Float32,64))

# a random domain vector with values in [-1,+1]
m = -1 .+ 2*rand(domain(F))
m0 = -1 .+ 2*rand(domain(F))

# Apply the nonlinear envelope operator to the domain vector m and return the result in the range vector d
d = F*m

# Gradient
g = gradient(x -> mse(F*x - d), m0)[1]

diff = norm(g - jacobian(F, m0)'*(F*m0 - d))
println("Difference between true and Flux gradient: ", diff)

```
"""
function ChainRulesCore.rrule(::typeof(*), F::T, x) where {T<:Jets.JopNl}
    y = F*x
    bck(Δy) = (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), Jets.jacobian(F, x)'*Δy)
    return y, bck
end

end