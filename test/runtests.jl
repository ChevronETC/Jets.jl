# TODO... turn this into actual unit-tests

using Revise

using JotNew, LinearAlgebra

A = JotOpDiagonal([3.0,4.0])

m = [2.0,3.0]

d_check = [6.0, 12.0]
d = A*m
d ≈ d_check
mul!(d, A, m)
d ≈ d_check

A₁ = JotOpDiagonal([2.0,3.0])
A₂ = JotOpDiagonal([4.0,5.0])

A = A₂ ∘ A₁
d = A*m
d_check = A₂*(A₁*m)
d ≈ d_check

A₃ = JotOpDiagonal([6.0,7.0])
A = A₃ ∘ A₂ ∘ A₁
d = A*m
d_check = A₃*(A₂*(A₁*m))
d ≈ d_check

A = [A₁ A₂ ; A₂ A₁]

A.ops[1,1]
A[1,1]

domain(A[1,1])
domain(A)

shape(A,1)
size(A,1)

m = [1.0,2.0,3.0,4.0]
length(m)

d = A*m
d_check = [A₁*m[1:2] + A₂*m[3:4] ; A₂*m[1:2] + A₁*m[3:4]]
d ≈ d_check

B = JotOpDiagonal([2.0,3.0,4.0,2.0])
C = B ∘ A
d = C*m

d_check = B*(A*m)
d ≈ d_check

A = [A₁∘A₂ A₁∘A₂ ; A₁∘A₂ A₁∘A₂]
domain(A)
m=rand(domain(A))
d = A*m
d_check = zeros(Float64,4)
d_check[1:2] .= A₁*(A₂*m[1:2]) .+ A₁*(A₂*m[3:4])
d_check[3:4] .= A₁*(A₂*m[1:2]) .+ A₁*(A₂*m[3:4])
d ≈ d_check

B₁ = A₁'

m = rand(domain(B₁))
d = B₁*m
d_check = A₁*m
d ≈ d_check

B = A'

domain(B)
m = rand(domain(B))
d = B*m
