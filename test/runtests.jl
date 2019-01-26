using Jets, LinearAlgebra, Test

function JopFoo(diag)
    df!(d,m;diagonal,kwargs...) = d .= diagonal .* m
    spc = JetSpace(Float64, length(diag))
    JopLn(;df! = df!, df′! = df!, dom = spc, rng = spc, s = (diagonal=diag,))
end

function JopBar(n)
    f!(d,m) = d .= m.^2
    df!(δd,δm;mₒ,kwargs...) = δd .= 2 .* mₒ .* δm
    spc = JetSpace(Float64, n)
    JopNl(f! = f!, df! = df!, df′! = df!, dom = spc, rng = spc)
end

function JopBaz(A)
    df!(d,m;A,kwargs...) = d .= A * m
    df′!(m,d;A,kwargs...) = m .= A' * d
    dom = JetSpace(eltype(A), size(A,2))
    rng = JetSpace(eltype(A), size(A,1))
    JopLn(;df! = df!, df′! = df′!, dom = dom, rng = rng, s = (A=A,))
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
    @test shape(✈, 1) == (10,2)
    @test shape(✈, 2) == (20,)
    @test shape(✈) == ((10,2),(20,))
    @test size(✈, 1) == 20
    @test size(✈, 2) == 20
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
    J = jacobian(F, m)
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

@testset "composition, linear" begin
    d₁,d₂,d₃,d₄ = map(i->rand(10), 1:4)
    A₁,A₂,A₃,A₄ = map(d->JopBaz(rand(10,10)), (d₁,d₂,d₃,d₄))
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

    a = A₂₁'*d
    @test a ≈ (B₁' * ( B₂' * d))
    a = A₃₂₁'*d
    @test a ≈ (B₁' * ( B₂' * ( B₃' * d )))
    a = A₄₃₂₁'*d
    @test a ≈ (B₁' * ( B₂' * ( B₃' * ( B₄' * d))))

    @test domain(A₄₃₂₁) == JetSpace(Float64, 10)
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

    m = rand(10)
    J₁ = jacobian(F₁, m)
    J₂₁ = jacobian(F₂, F₁*m) ∘ J₁
    J₃₂₁ = jacobian(F₃, (F₂ ∘ F₁) * m) ∘ jacobian(F₂, F₁ * m) ∘ J₁
    J₄₃₂₁ = jacobian(F₄, (F₃ ∘ F₂ ∘ F₁) * m) ∘ jacobian(F₃, (F₂ ∘ F₁) * m) ∘ jacobian(F₂, F₁ * m) ∘ J₁

    L₁ = jacobian(F₁, m)
    L₂₁ = jacobian(F₂₁, m)
    L₃₂₁ = jacobian(F₃₂₁, m)
    L₄₃₂₁ = jacobian(F₄₃₂₁, m)

    δm = rand(10)
    @test J₁ * δm ≈ L₁ * δm
    @test J₂₁ * δm ≈ L₂₁ * δm
    @test J₃₂₁ * δm ≈ L₃₂₁ * δm
    @test J₄₃₂₁ * δm ≈ L₄₃₂₁ * δm
end

@testset "composition, linear+nonlinear" begin
    d₁,d₄ = map(i->rand(10), 1:2)
    A₂,A₄ = map(d->JopFoo(d), (d₁,d₄))
    F₁,F₃ = map(i->JopBar(10), 1:2)
    F₂₁ = A₂ ∘ F₁
    F₃₂₁ = F₃ ∘ A₂ ∘ F₁
    F₄₃₂₁ = A₄ ∘ F₃ ∘ A₂ ∘ F₁
    m = rand(domain(F₁))
    d = F₂₁ * m
    @test d ≈ A₂*(F₁*m)
    d = F₃₂₁*m
    @test d ≈ F₃*( A₂ * ( F₁ * m))
    d = F₄₃₂₁ * m
    @test d ≈ A₄ * ( F₃ * ( A₂ * ( F₁ * m)))

    m = rand(10)
    J₁ = jacobian(F₁, m)
    J₂₁ = A₂ ∘ jacobian(F₁, m)
    J₃₂₁ = jacobian(F₃, A₂*(F₁*m)) ∘ A₂ ∘ jacobian(F₁, m)
    J₄₃₂₁ = A₄ ∘ jacobian(F₃, A₂*(F₁*m)) ∘ A₂ ∘ jacobian(F₁, m)

    L₁ = jacobian(F₁, m)
    L₂₁ = jacobian(F₂₁, m)
    L₃₂₁ = jacobian(F₃₂₁, m)
    L₄₃₂₁ = jacobian(F₄₃₂₁, m)

    δm = rand(10)
    @test J₁ * δm ≈ L₁ * δm
    @test J₂₁ * δm ≈ L₂₁ * δm
    @test J₃₂₁ * δm ≈ L₃₂₁ * δm
    @test J₄₃₂₁ * δm ≈ L₄₃₂₁ * δm
end

@testset "block operator" begin
    B₁₁,B₁₃,B₁₄,B₂₁,B₂₃,B₂₄,B₃₂,B₃₃ = map(i->rand(10,10), 1:8)
    A₁₁,A₁₃,A₁₄,A₂₁,A₂₃,A₂₄,A₃₂,A₃₃ = map(B->JopBaz(B), (B₁₁,B₁₃,B₁₄,B₂₁,B₂₃,B₂₄,B₃₂,B₃₃))
    F₁₂,F₂₃,F₃₁ = map(i->JopBar(10), 1:3)
    Z₂₂,Z₃₄ = map(i->JopZeroBlock(JetSpace(Float64,10), JetSpace(Float64,10)), 1:2)

    C₂₄ = A₂₄ ∘ JopBar(10)

    G = [A₁₁ F₁₂ A₁₃ A₁₄;
         A₂₁ Z₂₂ F₂₃ C₂₄;
         F₃₁ A₃₂ A₃₃ Z₃₄]

    @test isa(@blockop(G), Jets.JopBlock)

    F = @blockop [A₁₁ F₁₂ A₁₃ A₁₄;
                  A₂₁ Z₂₂ F₂₃ C₂₄;
                  F₃₁ A₃₂ A₃₃ Z₃₄]

    m = rand(domain(F))
    d = F*m
    @test d[1:10]  ≈ B₁₁ * m[1:10] + F₁₂ * m[11:20] + B₁₃ * m[21:30] + B₁₄ * m[31:40]
    @test d[11:20] ≈ B₂₁ * m[1:10]                  + F₂₃ * m[21:30] + C₂₄ * m[31:40]
    @test d[21:30] ≈ F₃₁ * m[1:10] + B₃₂ * m[11:20] + B₃₃ * m[21:30]

    J = jacobian(F,m)
    δm = rand(domain(J))
    δd = J * δm

    J₁₂ = jacobian(F₁₂,m[11:20])
    J₂₃ = jacobian(F₂₃,m[21:30])
    J₂₄ = jacobian(C₂₄,m[31:40])
    J₃₁ = jacobian(F₃₁,m[1:10])

    L = @blockop [A₁₁ J₁₂ A₁₃ A₁₄;
                  A₂₁ Z₂₂ J₂₃ J₂₄;
                  J₃₁ A₃₂ A₃₃ Z₃₄]

    @test δd ≈ L*δm
    @test L'*δd ≈ J'*δd

    @test getblockrange(δd, L, 2) ≈ δd[11:20]
    setblockrange!(δd, L, 2, π*ones(10))
    @test getblockrange(δd, L, 2) ≈ π*ones(10)

    @test getblockrange(d, F, 2) ≈ d[11:20]
    setblockrange!(d, F, 2, π*ones(10))
    @test getblockrange(d, F, 2) ≈ π*ones(10)

    @test getblockdomain(δm, L, 2) ≈ δm[11:20]
    setblockdomain!(δm, L, 2, π*ones(10))
    @test getblockdomain(δm, L, 2) ≈ π*ones(10)

    @test getblockdomain(m, F, 2) ≈ m[11:20]
    setblockdomain!(m, F, 2, π*ones(10))
    @test getblockdomain(m, F, 2) ≈ π*ones(10)
end
