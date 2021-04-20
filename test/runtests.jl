using Jets, LinearAlgebra, Random, Test

JopFoo_df!(d,m;diagonal,kwargs...) = d .= diagonal .* m
function JopFoo(diag)
    spc = JetSpace(Float64, length(diag))
    JopLn(;df! = JopFoo_df!, dom = spc, rng = spc, s = (diagonal=diag,))
end
Jets.perfstat(J::Jet{D,R,typeof(JopFoo_df!)}) where {D,R} = Float64(π)

JopClose_df!(d,m;diagonal,kwargs...) = d .= diagonal .* m
function JopClose(diag)
    spc = JetSpace(Float64, length(diag))
    file = tempname()
    touch(file)
    JopLn(;df! = JopClose_df!, dom = spc, rng = spc, s = (diagonal=diag, file=file))
end
Base.close(J::Jet{D,R,typeof(JopClose_df!)}) where {D,R} = rm(state(J).file)

JopBar_f!(d,m;kwargs...) = d .= m.^2
JopBar_df!(δd,δm;mₒ,kwargs...) = δd .= 2 .* mₒ .* δm
function JopBar(n)
    spc = JetSpace(Float64, n)
    JopNl(f! = JopBar_f!, df! = JopBar_df!, dom = spc, rng = spc)
end

JopBaz_df!(d,m;A,kwargs...) =  d .= A * m
JopBaz_df′!(m,d;A,kwargs...) = m .= A' * d
function JopBaz(A)
    dom = JetSpace(eltype(A), size(A,2))
    rng = JetSpace(eltype(A), size(A,1))
    JopLn(;df! = JopBaz_df!, df′! = JopBaz_df′!, dom = dom, rng = rng, s = (A=A,))
end

JopFooBar_df!(d,m;A,kwargs...) =  d .= A.*m
function JopFooBar(n)
    spc = JetSpace(Float64,n,n)
    JopLn(df! = JopFooBar_df!, dom = spc, rng = spc, s= (A=rand(n,n),))
end

JopRosenbrock_f!(d,m;kwargs...) = d.= [1-m[1], 10*(m[2]-m[1]^2)]
JopRosenbrock_df!(d,m;J,kwargs...) = d .= J*m
JopRosenbrock_df′!(m,d;J,kwargs...) = m .= J'*d
JopRosenbrock_upstate!(m,s) = s.J[2,1] = -20.0*m[1]
function JopRosenbrock()
    dom = JetSpace(Float64, 2)
    rng = JetSpace(Float64, 2)
    JopNl(f! = JopRosenbrock_f!, df! = JopRosenbrock_df!, df′! = JopRosenbrock_df′!,
        upstate! = JopRosenbrock_upstate!, dom = dom, rng = rng, s = (J=[-1.0 0.0;0.0 10.0],))
end

JopFoo2_df!(d,m;diagonal,kwargs...) = d .= diagonal .* m
function JopFoo2(diag)
    spc = JetSpace(Float64, size(diag))
    JopLn(;df! = JopFoo2_df!, dom = spc, rng = spc, s = (diagonal=diag,))
end

@testset "JetSpace, construction, n=$n, T=$T" for n in ((2,),(2,3),(2,3,4)), T in (Float32,Float64,Complex{Float32},Complex{Float64})
    N = length(n)
    R = JetSpace(T, n...)
    @test size(R) == n
    @test eltype(R) == T
    @test eltype(typeof(R)) == T
    @test ndims(R) == N
end

@testset "JetSpace, operations, n=$n, T=$T" for n in ((2,),(2,3),(2,3,4)), T in (Float32,Float64,Complex{Float32},Complex{Float64})
    R = JetSpace(T, n...)
    N = length(n)
    x = rand(R)
    @test eltype(x) == T
    @test size(x) == n
    @test ndims(R) == N
    y = rand(T, n)
    z = copy(y)[:]
    @test y ≈ reshape(z,  R)
    @test ones(R) ≈ ones(T, n)
    @test size(rand(R)) == size(R)
    @test zeros(R) ≈ zeros(T, n)
    @test size(Array(R)) == size(R)
    @test length(randperm(R, 2)) == 2
    @test typeof(randperm(R, 2)) == Vector{Int}
end

@testset "Jet, construction" begin
    f!(d,m;a) = d .= a .* m.^2
    df!(δd,δm;a,mₒ) = δd .= 2 .* a .* mₒ .* δm
    a = rand(20)
    ✈ = Jet(;
        dom = JetSpace(Float64,20),
        rng = JetSpace(Float64,10,2),
        f! = f!,
        df! = df!,
        df′! = df!,
        s = (a=a,))
    @test domain(✈) == JetSpace(Float64,20)
    @test range(✈) == JetSpace(Float64,10,2)
    @test point(✈) ≈ zeros(eltype(domain(✈)),ntuple(i->0,ndims(domain(✈))))
    mₒ = rand(domain(✈))
    Jets.point!(✈, mₒ)
    @test point(✈) ≈ mₒ
    a .= rand(20)
    state!(✈, (a=a,))
    s = state(✈)
    @test s.a ≈ a
    @test state(✈, :a) ≈ a
    @test shape(✈, 1) == (10,2)
    @test shape(✈, 2) == (20,)
    @test shape(✈) == ((10,2),(20,))
    @test size(✈, 1) == 20
    @test size(✈, 2) == 20
    @test close(✈) == false
end

@testset "linear operator" begin
    diag = rand(10)
    A = JopFoo(diag)
    m = rand(10)
    d = A*m
    @test d ≈ diag .* m
    a = A'*d
    @test a ≈ diag .* d

    d .= 0
    mul!(d, A, m)
    @test d ≈ diag .* m
    a .= 0
    mul!(a, A', d)
    @test a ≈ diag .* d

    @test size(A) == (10,10)
    @test shape(A) == ((10,), (10,))
    @test size(A,1) == 10
    @test size(A,2) == 10
    @test shape(A,1) == (10,)
    @test shape(A,2) == (10,)
    @test domain(A) == JetSpace(Float64,10)
    @test range(A) == JetSpace(Float64,10)
    @test eltype(A) == Float64
    @test convert(Array, A) ≈ diagm(0=>diag)

    @test state(A).diagonal ≈ diag
    @test state(A, :diagonal) ≈ diag

    B = JopFooBar(5)
    @test convert(Array, B) ≈ diagm(0=>vec(state(B).A))
end

@testset "nonlinear operator" begin
    n = 10
    F = JopBar(n)
    m = rand(domain(F))
    d = F*m
    @test d ≈ m.^2
    d .= 0
    mul!(d, F, m)
    @test d ≈ m.^2
    J = jacobian!(F, m)
    @test point(J) ≈ m
    d = J*m
    @test d ≈ 2 .* point(J) .* m
    a = J'*d
    @test a ≈ 2 .* point(J) .* d

    @test size(F) == (10,10)
    @test shape(F) == ((10,), (10,))
    @test size(F,1) == 10
    @test size(F,2) == 10
    @test shape(F,1) == (10,)
    @test shape(F,2) == (10,)
    @test domain(F) == JetSpace(Float64,10)
    @test range(F) == JetSpace(Float64,10)
end

@testset "upstate" begin
    F = JopRosenbrock()
    m = rand(2)
    J = jacobian!(F, m)
    @test state(J).J ≈ [-1.0 0.0;-20*m[1] 10.0]
end

@testset "multiple simultaneous linearizations" begin
    F = JopBar(2)
    J₁ = jacobian(F, [1.0,2.0])
    J₂ = jacobian(F, [3.0,4.0])

    δm = [1.0, 2.0]
    @test J₁ * δm ≈ 2 .* [1.0,2.0] .* δm
    @test J₂ * δm ≈ 2 .* [3.0,4.0] .* δm

    J₁ = jacobian!(F, [1.0,2.0])
    J₂ = jacobian!(F, [3.0,4.0])

    @test J₁ * δm ≈ 2 .* [3.0,4.0] .* δm
    @test J₂ * δm ≈ J₁ * δm
end

function indexmap(I)
    if I[1] < 5
        return CartesianIndex(I)
    else
        return CartesianIndex((I[1]-4,I[2]))
    end
end

@testset "Symmetric space" begin
    R = Jets.JetSSpace(Complex{Float64}, (8,4), (4,4), indexmap)
    @test size(R) == (8,4)
    @test eltype(R) == Complex{Float64}
    real.(ones(R)) ≈ real.(ones(Complex{Float64},8,4))
    imag.(ones(R)) ≈ imag.(ones(Complex{Float64},8,4))
    @test real.(zeros(R)) ≈ zeros(8,4)
    @test imag.(zeros(R)) ≈ zeros(8,4)
    @test size(rand(R)) == (8,4)
    @test size(Array(R)) == (8,4)
    @test eltype(Array(R)) == Complex{Float64}
    x = rand(R)
    z = similar(x)
    @test typeof(z) == Jets.SymmetricArray{Complex{Float64},2,typeof(indexmap)}
    y = x.A
    @test norm(x) ≈ sqrt(2*(norm(y)^2))
    @test norm(x,2) ≈ norm(x)
    @test norm(x,1) ≈ 2*norm(y,1)
    @test norm(x,Inf) ≈ norm(y,Inf)
    x[1,1] = x[6,1] = 0
    @test norm(x,0) ≈ 2*norm(y,0)
end

@testset "Symmetric spaces, broadcast" begin
    R = Jets.JetSSpace(Complex{Float64}, (8,4), (4,4), indexmap)
    u = rand(R)
    v = rand(R)
    w = rand(R)
    a = rand(Float64)
    b = rand(Float64)
    c = rand(Float64)
    x = a*u .+ b*v .+ c*w
    @test typeof(x) == Jets.SymmetricArray{Complex{Float64},2,typeof(indexmap)}

    for i2 = 1:4, i1 = 1:4
        @test x.A[i1,i2] ≈ a*u.A[i1,i2] + b*v.A[i1,i2] + c*w.A[i1,i2]
    end

    y = zeros(R)
    y .= x
    @test typeof(x) == Jets.SymmetricArray{Complex{Float64},2,typeof(indexmap)}

    for i2 = 1:4, i1 = 1:4
        @test y.A[i1,i2] ≈ a*u.A[i1,i2] + b*v.A[i1,i2] + c*w.A[i1,i2]
    end

    y .= x
    @test y ≈ x
    y .*= -1
    for i = 1:length(parent(y))
        parent(y)[i] ≈ -parent(x)[i]
    end

    z = abs.(x)
    @test typeof(z) == Array{Float64,2}
    for i = 1:length(z)
        @test z[i] ≈ abs(x[i])
    end
end

@testset "composition, linear" begin
    A₁,A₂,A₃,A₄ = map(d->JopBaz(rand(10,10)), 1:4)
    B₁,B₂,B₃,B₄ = map(A->state(A).A, (A₁,A₂,A₃,A₄))
    A₂₁ = A₂ ∘ A₁
    A₃₂₁ = A₃ ∘ A₂ ∘ A₁
    A₄₃₂₁ = A₄ ∘ A₃ ∘ A₂ ∘ A₁
    m = rand(domain(A₁))
    d = A₂₁*m
    @test d ≈ B₂ * ( B₁ * m )
    d = A₃₂₁*m
    @test d ≈ B₃ * (B₂ * ( B₁ * m))
    d = A₄₃₂₁*m
    @test d ≈ B₄ * (B₃ * ( B₂ * ( B₁ * m)))
    @test length(state(A₄₃₂₁).ops) == 4

    a = A₂₁'*d
    @test a ≈ (B₁' * ( B₂' * d))
    a = A₃₂₁'*d
    @test a ≈ (B₁' * ( B₂' * ( B₃' * d )))
    a = A₄₃₂₁'*d
    @test a ≈ (B₁' * ( B₂' * ( B₃' * ( B₄' * d))))

    @test domain(A₄₃₂₁) == JetSpace(Float64, 10)
    @test eltype(A₄₃₂₁) == Float64

    B₄₃₂₁ = convert(Array, A₄₃₂₁)
    @test B₄₃₂₁*m ≈ A₄₃₂₁*m

    C₄₃₂₁ = A₄ ∘ A₃ ∘ A₂₁'
    @test C₄₃₂₁*m ≈ A₄ * (A₃ * (A₁' * (A₂' * m)))
end

@testset "composition, linear with matrices" begin
    A₁,A₂,A₄ = map(d->JopBaz(rand(10,10)), 1:4)
    A₃ = rand(10,10)
    B₁,B₂,B₄ = map(A->state(A).A, (A₁,A₂,A₄))
    B₃ = A₃
    A₂₁ = A₂ ∘ A₁
    A₃₂₁ = A₃ ∘ A₂ ∘ A₁
    A₄₃₂₁ = A₄ ∘ A₃ ∘ A₂ ∘ A₁
    m = rand(domain(A₁))
    d = A₂₁*m
    @test d ≈ B₂ * ( B₁ * m )
    d = A₃₂₁*m
    @test d ≈ B₃ * (B₂ * ( B₁ * m))
    d = A₄₃₂₁*m
    @test d ≈ B₄ * (B₃ * ( B₂ * ( B₁ * m)))

    a = A₂₁'*d
    @test a ≈ (B₁' * ( B₂' * d))
    a = A₃₂₁'*d
    @test a ≈ (B₁' * ( B₂' * ( B₃' * d )))
    a = A₄₃₂₁'*d
    @test a ≈ (B₁' * ( B₂' * ( B₃' * ( B₄' * d))))

    @test domain(A₄₃₂₁) == JetSpace(Float64, 10)
    @test eltype(A₄₃₂₁) == Float64

    B₄₃₂₁ = convert(Array, A₄₃₂₁)
    @test B₄₃₂₁*m ≈ A₄₃₂₁*m
end

@testset "composition, nonlinear" begin
    F₁,F₂,F₃,F₄ = map(i->JopBar(10), 1:4)
    F₂₁ = F₂ ∘ F₁
    F₃₂₁ = F₃ ∘ F₂ ∘ F₁
    F₄₃₂₁ = F₄ ∘ F₃ ∘ F₂ ∘ F₁
    m = rand(domain(F₁))
    d = F₂₁ * m
    @test d ≈ F₂*(F₁*m)
    d = F₃₂₁*m
    @test d ≈ F₃*( F₂ * ( F₁ * m))
    d = F₄₃₂₁ * m
    @test d ≈ F₄ * ( F₃ * ( F₂ * ( F₁ * m)))

    m = ones(10)
    J₁ = jacobian!(F₁, m)
    J₂₁ = jacobian!(F₂, F₁*m) ∘ J₁
    J₃₂₁ = jacobian!(F₃, (F₂ ∘ F₁) * m) ∘ jacobian!(F₂, F₁ * m) ∘ J₁
    J₄₃₂₁ = jacobian!(F₄, (F₃ ∘ F₂ ∘ F₁) * m) ∘ jacobian!(F₃, (F₂ ∘ F₁) * m) ∘ jacobian!(F₂, F₁ * m) ∘ J₁

    L₁ = jacobian!(F₁, m)
    L₂₁ = jacobian!(F₂₁, m)
    L₃₂₁ = jacobian!(F₃₂₁, m)
    L₄₃₂₁ = jacobian!(F₄₃₂₁, m)

    δm = ones(10)
    @test J₁ * δm ≈ L₁ * δm
    @test J₂₁ * δm ≈ L₂₁ * δm
    @test J₃₂₁ * δm ≈ L₃₂₁ * δm
    @test J₄₃₂₁ * δm ≈ L₄₃₂₁ * δm

    δd = J₄₃₂₁*δm
    @test J₄₃₂₁'*δd ≈ L₄₃₂₁'*δd
end

@testset "composition, linear+nonlinear, adjoints" begin
    A₂ = JopBaz(rand(10,10))
    A₄ = JopFoo(rand(10))
    F₁,F₃ = map(i->JopBar(10), 1:2)
    F₂₁ = A₂' ∘ F₁
    F₃₂₁ = F₃ ∘ A₂' ∘ F₁
    F₄₃₂₁ = A₄ ∘ F₃ ∘ A₂' ∘ F₁
    m = rand(domain(F₁))
    d = F₂₁ * m
    @test d ≈ A₂'*(F₁*m)
    d = F₃₂₁*m
    @test d ≈ F₃*( A₂' * ( F₁ * m))
    d = F₄₃₂₁ * m
    @test d ≈ A₄ * ( F₃ * ( A₂' * ( F₁ * m)))

    m = rand(10)
    J₁ = jacobian!(F₁, m)
    J₂₁ = A₂' ∘ jacobian!(F₁, m)
    J₃₂₁ = jacobian!(F₃, A₂'*(F₁*m)) ∘ A₂' ∘ jacobian!(F₁, m)
    J₄₃₂₁ = A₄ ∘ jacobian!(F₃, A₂'*(F₁*m)) ∘ A₂' ∘ jacobian!(F₁, m)

    L₁ = jacobian!(F₁, m)
    L₂₁ = jacobian!(F₂₁, m)
    L₃₂₁ = jacobian!(F₃₂₁, m)
    L₄₃₂₁ = jacobian!(F₄₃₂₁, m)

    δm = rand(10)
    @test J₁ * δm ≈ L₁ * δm
    @test J₂₁ * δm ≈ L₂₁ * δm
    @test J₃₂₁ * δm ≈ L₃₂₁ * δm
    @test J₄₃₂₁ * δm ≈ L₄₃₂₁ * δm
end

@testset "composition operator * operator" begin
    A₁ = JopFoo(rand(2))
    A₂ = @blockop [JopBar(2), JopBar(2)]
    A = A₂ ∘ A₁

    A₁₁ = getblock(A, 1, 1)
    A₂₁ = getblock(A, 2, 1)

    m = rand(domain(A))
    @test getblock(A * m, 1) ≈ A₁₁ * m
    @test getblock(A * m, 2) ≈ A₂₁ * m
end

@testset "composition, state" begin
    diag = rand(2,2)
    A = JopFoo(diag)
    F = JopBar(2)
    G = A ∘ F
    @test state(G, :diagonal) ≈ state(A, :diagonal)
    @test state(G, :diagonal) ≈ state(A).diagonal

    @test_throws Exception state(G, :foo)

    B = JopFoo(diag)
    C = A ∘ B
    @test_throws Exception state(C, :digaonal)
end

@testset "sum, linear" begin
    A₁,A₂,A₃ = map(i->JopBaz(rand(10,10)), 1:3)
    A₁₂ = A₁ + A₂
    A₁₂₃ = A₁ + A₂ - A₃

    m = rand(domain(A₁))
    @test A₁₂ * m ≈ A₁*m + A₂*m
    @test A₁₂₃ * m ≈ A₁*m + A₂*m - A₃*m

    A₁₂₃ = A₁₂ + A₃
    @test A₁₂₃ * m ≈ A₁*m + A₂*m + A₃*m
    A₁₂₃₁₂ = A₁₂₃ - A₁₂
    @test A₁₂₃₁₂*m ≈ A₁*m + A₂*m + A₃*m - A₁*m - A₂*m

    d = rand(range(A₁))
    @test A₁₂₃' * d ≈ A₁'*d + A₂'*d + A₃'*d
end

@testset "sum, linear with compositions" begin
    A₁,A₂,A₃ = map(i->JopBaz(rand(10,10)), 1:3)
    a₁,a₂,a₃ = map(i->rand(), 1:3)
    A₁₂ = a₁*A₁ + a₂*A₂
    A₁₂₃ = a₁*A₁ + a₂*A₂ - a₃*A₃

    m = rand(domain(A₁))
    @test A₁₂ * m ≈ a₁*(A₁*m) + a₂*(A₂*m)
    @test A₁₂₃ * m ≈ a₁*(A₁*m) + a₂*(A₂*m) - a₃*(A₃*m)

    A₁₂₃ = A₁₂ + a₃*A₃
    @test A₁₂₃ * m ≈ a₁*(A₁*m) + a₂*(A₂*m) + a₃*(A₃*m)
    A₁₂₃₁₂ = A₁₂₃ - A₁₂
    @test A₁₂₃₁₂*m ≈ a₁*(A₁*m) + a₂*(A₂*m) + a₃*(A₃*m) - a₁*(A₁*m) - a₂*(A₂*m)

    d = rand(range(A₁))
    @test A₁₂₃' * d ≈ a₁*(A₁'*d) + a₂*(A₂'*d) + a₃*(A₃'*d)
end

@testset "sum, linear with matrices" begin
    A₁,A₃ = map(i->JopBaz(rand(10,10)), 1:3)
    A₂ = rand(10,10)
    A₁₂ = A₁ + A₂
    A₁₂₃ = A₁ + A₂ - A₃
    m = rand(domain(A₁))
    @test A₁₂ * m ≈ A₁*m + A₂*m
    @test A₁₂₃ * m ≈ A₁*m + A₂*m - A₃*m
end

@testset "sum, linear+nonlinar" begin
    A₁ = JopBaz(rand(10,10))
    F₂ = JopBar(10)

    F₁₂ = A₁ + F₂
    m = rand(domain(F₁₂))
    @test F₁₂*m ≈ A₁*m + F₂*m

    J₁₂ = jacobian(F₁₂,m)
    @test J₁₂*m ≈ A₁*m + jacobian(F₂,m)*m
end

@testset "block arrays" begin
    R = Jets.JetBSpace([JetSpace(Float64,2),JetSpace(Float64,2,2),JetSpace(Float64,2,3)])
    x = ones(R)
    @test getblock(x,1) ≈ ones(2)
    @test getblock(x,2) ≈ ones(2,2)
    @test getblock(x,3) ≈ ones(2,3)
    setblock!(x,1,π)
    setblock!(x,2,2π)
    setblock!(x,3,3*pi*ones(2,3))
    @test getblock!(x, 2, Array{Float64}(undef,2,2)) ≈ 2π*ones(2,2)
    _x = convert(Array, x)
    @test _x ≈ x
    @test norm(x) ≈ norm(_x)
    @test norm(x,0) ≈ norm(_x,0)
    @test norm(x,Inf) ≈ norm(_x,Inf)

    x = rand(R)
    _x = convert(Array, x)
    mn,mx = extrema(x)
    _mn,_mx = extrema(_x)
    @test mn ≈ _mn
    @test mx ≈ _mx

    R = Jets.JetBSpace([JetSpace(ComplexF64,2),JetSpace(ComplexF64,2,2),JetSpace(ComplexF64,2,3)])
    x = rand(R)
    _x = convert(Array, x)
    y = abs.(x)
    @test eltype(y) == Float64
    @test y ≈ abs.(_x)
end

@testset "block arrays, broadcasting" begin
    R = Jets.JetBSpace([JetSpace(Float64,2),JetSpace(Float64,2,2),JetSpace(Float64,2,3)])
    u = rand(R)
    v = rand(R)
    w = rand(R)
    a = rand(Float64)
    b = rand(Float64)
    c = rand(Float64)
    x = a*u .+ b*v .+ c*w
    @test typeof(x) == Jets.BlockArray{Float64,Array{Float64}}

    for i = 1:length(x.arrays)
        for j = 1:length(x.arrays[i])
            @test x.arrays[i][j] ≈ a*u.arrays[i][j] + b*v.arrays[i][j] + c*w.arrays[i][j]
        end
    end

    y = zeros(R)
    y .= x
    @test typeof(y) == Jets.BlockArray{Float64,Array{Float64}}

    for i = 1:length(x.arrays)
        for j = 1:length(x.arrays[i])
            @test y.arrays[i][j] ≈ a*u.arrays[i][j] + b*v.arrays[i][j] + c*w.arrays[i][j]
        end
    end

    z = zeros(12)
    z .= x
    @test typeof(z) == Array{Float64,1}
    @test z ≈ x
    x .= 3.14
    for i = 1:length(x)
        @test x[i] ≈ 3.14
    end

    x = rand(Int32, size(R))
    y = rand(R)
    z = x .* y
    @test isa(z, Jets.BlockArray)
    @test convert(Array, z) ≈ x .* convert(Array, y)

    x = similar(y, Int)
end

@testset "block array, reshaped from array" begin
    x = rand(5,10)
    R = JetBSpace([JetSpace(Float64,5) for i=1:10])
    _x = reshape(x, R)
    for i = 1:length(_x)
        @test _x[i] ≈ x[i]
        _x[i] = i
        @test x[i] ≈ i
    end
    A = [rand(10,10) for i = 1:5]
    _A = @blockop [JopBaz(A[i]) for i = 1:5]

    m = rand(domain(_A))
    _y = _A*m
    for i = 1:5
        @test getblock(_y, i) ≈ A[i]*m
    end
end

@testset "block operator" begin
    B₁₁,B₁₃,B₁₄,B₂₁,B₂₃,B₂₄,B₃₂,B₃₃ = map(i->rand(10,10), 1:8)
    A₁₁,A₁₃,A₁₄,A₂₁,A₂₃,A₃₂,A₃₃ = map(B->JopBaz(B), (B₁₁,B₁₃,B₁₄,B₂₁,B₂₃,B₃₂,B₃₃))
    A₂₄ = (JopBaz(B₂₄))'
    F₁₂,F₂₃,F₃₁ = map(i->JopBar(10), 1:3)
    Z₂₂,Z₃₄ = map(i->JopZeroBlock(JetSpace(Float64,10), JetSpace(Float64,10)), 1:2)

    @test iszero(Z₂₂)
    @test !iszero(A₁₁)
    @test !iszero(F₁₂)

    C₂₄ = A₂₄ ∘ JopBar(10)

    G = [A₁₁ F₁₂ A₁₃ A₁₄;
         A₂₁ Z₂₂ F₂₃ C₂₄;
         F₃₁ A₃₂ A₃₃ Z₃₄]

    @test isa(@blockop(G), JopNl{<:Jet{<:Jets.JetBSpace,<:Jets.JetBSpace,typeof(Jets.JetBlock_f!)}})

    F = @blockop [A₁₁ F₁₂ A₁₃ A₁₄;
                  A₂₁ Z₂₂ F₂₃ C₂₄;
                  F₃₁ A₃₂ A₃₃ Z₃₄]

    @test nblocks(F) == (3,4)
    @test nblocks(F,1) == 3
    @test nblocks(F,2) == 4
    @test nblocks(A₁₁) == (1,1)

    _F₁₂ = getblock(JopNl,F,1,2)
    m = rand(domain(F₁₂))
    @test F₁₂*m ≈ _F₁₂*m

    _A₁₁ = getblock(JopLn,F,1,1)
    m = rand(domain(A₁₁))
    @test A₁₁*m ≈ _A₁₁*m

    @test ones(domain(F)) ≈ ones(40)
    @test zeros(range(F)) ≈ zeros(30)
    @test size(rand(range(F))) == (30,)

    m = rand(domain(F))
    d = F*m
    @test d[1:10]  ≈ B₁₁ * m[1:10] + F₁₂ * m[11:20] + B₁₃ * m[21:30] + B₁₄ * m[31:40]
    @test d[11:20] ≈ B₂₁ * m[1:10]                  + F₂₃ * m[21:30] + C₂₄ * m[31:40]
    @test d[21:30] ≈ F₃₁ * m[1:10] + B₃₂ * m[11:20] + B₃₃ * m[21:30]

    J = jacobian!(F,m)
    δm = rand(domain(J))
    δd = J * δm

    J₁₂ = jacobian!(F₁₂,m[11:20])
    J₂₃ = jacobian!(F₂₃,m[21:30])
    J₂₄ = jacobian!(C₂₄,m[31:40])
    J₃₁ = jacobian!(F₃₁,m[1:10])

    L = @blockop [A₁₁ J₁₂ A₁₃ A₁₄;
                  A₂₁ Z₂₂ J₂₃ J₂₄;
                  J₃₁ A₃₂ A₃₃ Z₃₄]

    @test δd ≈ L*δm
    @test L'*δd ≈ J'*δd

    @test mul!(rand(domain(L)), L', δd) ≈ J'*δd

    @test eltype(L) == Float64
    K = convert(Array, L)

    @test L*δm ≈ K*δm

    _J₁₂ = getblock(J, 1, 2)
    @test isa(J₁₂, JopLn)
    δm = rand(domain(J₁₂))
    @test J₁₂*δm ≈ _J₁₂*δm
end

@testset begin "block operator with keyword arguments"
    x = @blockop [JopBaz(rand(2,2)) ; JopBaz(rand(2,2))] foo = 3
    @test isa(x, Jop)
    x = @blockop [JopBaz(rand(2,2)) ; JopBaz(rand(2,2))] foo = 3 bar = 4
    @test isa(x, Jop)
end

@testset "block operator, singleton" begin
    B = JopBaz(rand(5,5))
    A = @blockop [B for i=1:1, j=1:1]
    m = rand(domain(A))
    @test A*m ≈ B*m
    d = rand(range(A))
    @test A'*d ≈ B'*d

    F = JopBar(5)
    G = @blockop [F for i=1:1, j=1:1]
    @test F*m ≈ G*m
    J = jacobian!(F, m)
    @test J*m ≈ jacobian!(G,m)*m
    @test J'*d ≈ (jacobian!(G,m)'*d)
end

@testset "block operator, tall-and-skinny" begin
    B = [JopBaz(rand(5,5)) for i=1:3, j=1:1]
    A = @blockop B
    m = rand(domain(A))
    @test A*m ≈ [B[1,1]*m;B[2,1]*m;B[3,1]*m]
    d = rand(range(A))
    @test A'*d ≈ B[1,1]'*d[1:5] + B[2,1]'*d[6:10] + B[3,1]'*d[11:15]

    G = [JopBar(5) for i=1:3, j=1:1]
    F = @blockop G
    @test F*m ≈ [G[1,1]*m;G[2,1]*m;G[3,1]*m]
    J = jacobian!(F,m)
    @test J*m ≈ [jacobian!(G[1,1],m)*m;jacobian!(G[2,1],m)*m;jacobian!(G[3,1],m)*m]
    @test J'*d ≈ jacobian!(G[1,1],m)'*d[1:5] + jacobian!(G[2,1],m)'*d[6:10] + jacobian!(G[3,1],m)'*d[11:15]

    B₁₁ = getblock(A,1,1)
    @test isa(B₁₁,JopLn)
    @test B₁₁*m ≈ B[1,1]*m

    @test nblocks(A) == (3,1)
    @test nblocks(A, 1) == 3
    @test nblocks(A, 2) == 1
end

@testset "block operator, short-and-fat" begin
    B = [JopBaz(rand(5,5)) for i=1:1, j=1:3]
    A = @blockop B
    m = rand(domain(A))
    @test A*m ≈ B[1,1]*m[1:5] + B[1,2]*m[6:10] + B[1,3]*m[11:15]
    d = rand(range(A))
    @test A'*d ≈ [B[1,1]'*d;B[1,2]'*d;B[1,3]'*d]

    G = [JopBar(5) for i=1:1, j=1:3]
    F = @blockop G
    @test F*m ≈ G[1,1]*m[1:5] + G[1,2]*m[6:10] + G[1,3]*m[11:15]
    J = jacobian!(F,m)
    @test J*m ≈ jacobian!(G[1,1],m[1:5])*m[1:5] + jacobian!(G[1,2],m[6:10])*m[6:10] + jacobian!(G[1,3],m[11:15])*m[11:15]
    @test J'*d ≈ [jacobian!(G[1,1],m[1:5])'*d;jacobian!(G[1,2],m[6:10])'*d;jacobian!(G[1,3],m[11:15])'*d]
end

@testset "getblock, adjoint operator" begin
    B = [JopBaz(rand(5,5)) for i=1:2, j=1:3]
    A = @blockop B
    C = A'

    for i = 1:2, j = 1:3
        Cji = getblock(C, j, i)
        @test isa(Cji, JopAdjoint)
        @test state(Cji).A ≈ state(B[i,j]).A
        m = rand(domain(Cji))
        @test Cji*m ≈ B[i,j]'*m
    end

    B = [JopBaz(rand(5,5))' for i=1:2, j=1:3]
    A = @blockop B
    C = A'

    C11 = getblock(C,1,1)
    isa(C11,JopLn)

    for i = 1:2, j=1:3
        Cji = getblock(C, j, i)
        @test isa(Cji, JopLn)
        @test state(Cji).A ≈ state(B[i,j]).A
        m = rand(domain(Cji))
        @test Cji*m ≈ B[i,j]'*m
    end
end

@testset "scalar multplying an operator" begin
    A = JopBaz(rand(10,10))
    a = 3.14
    m = rand(domain(A))
    B = a*A
    @test B*m ≈ a*(A*m)
end

@testset "vectorized operator" begin
    A = JopFoo2(rand(10,11))
    x = rand(domain(A))
    
    @test size(vec(domain(A))) == (10*11,)
    
    B = vec(A)
    @test isa(jet(B), Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(Jets.JetVec_f!)})
    
    d = A * x
    _d = vec(A) * vec(x)
    __d = reshape(_d, range(A))
    @test d[:] ≈ _d
    @test __d ≈ d
    @test size(d) == (10,11)
    @test size(_d) == (10*11,)
    @test size(__d) == (10,11)

    C = JopFoo(rand(10,11))
    @test isa(jet(C), Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(JopFoo_df!)})
end

@testset "vectorized block operator" begin
    A = @blockop [JopFoo2(rand(10,11)), JopFoo2(rand(10,11))]
    x = rand(domain(A))

    @test size(vec(domain(A))) == (10*11,)
    @test size(vec(range(A))) == (2*10*11,)
    @test size(range(A)) == size(vec(range(A)))

    B = vec(A)
    @test isa(jet(B), Jet{<:JetAbstractSpace,<:JetAbstractSpace,typeof(Jets.JetVec_f!)})

    d = A * x
    _d = vec(A) * vec(x)
    @test d ≈ _d

    a = A' * d
    _a = vec(A') * d
    __a = reshape(a, domain(A))
    @test a ≈ __a
end

@testset "close op" begin
    A = JopClose(rand(2))
    @test isfile(state(A).file)
    close(A)
    @test !isfile(state(A).file)
end

@testset "close block op" begin
    A = @blockop [JopClose(rand(2)) JopClose(rand(2)) ; JopClose(rand(2)) JopClose(rand(2))]
    A₁₁ = getblock(A,1,1)
    A₁₂ = getblock(A,1,2)
    A₂₁ = getblock(A,2,1)
    A₂₂ = getblock(A,2,2)
    for B in (A₁₁, A₁₂, A₂₁, A₂₂)
        @test isfile(state(B).file)
    end
    close(A)
    for B in (A₁₁, A₁₂, A₂₁, A₂₂)
        @test !isfile(state(B).file)
    end
end

@testset "close composite op" begin
    A₁ = JopClose(rand(2))
    A₂ = JopClose(rand(2))
    A = A₂ ∘ A₁
    for B in (A₁, A₂)
        @test isfile(state(B).file)
    end
    close(A)
    for B in (A₁, A₂)
        @test !isfile(state(B).file)
    end
end

@testset "perfstat" begin
    A₁ = JopFoo(rand(2))
    A₂ = JopBar(2)
    A = A₂ ∘ A₁

    @test perfstat(A₁) ≈ Float64(π)
    @test perfstat(A₂) == nothing
    @test perfstat(A) ≈ Float64(π)

    B = A₂ + A₁
    @test perfstat(B) ≈ Float64(π)
end
