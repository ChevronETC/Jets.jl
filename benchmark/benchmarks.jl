using BenchmarkTools, Jets, LinearAlgebra

const SUITE = BenchmarkGroup()

s = JetSpace(Float64, 100, 200)
x = rand(20_000)
SUITE["JetSpace"] = BenchmarkGroup()
SUITE["JetSpace"]["construct"] = @benchmarkable JetSpace(Float64, 2, 3)
SUITE["JetSpace"]["size"] = @benchmarkable size($s)
SUITE["JetSpace"]["eltype"] = @benchmarkable eltype($s)
SUITE["JetSpace"]["ndims"] = @benchmarkable ndims($s)
SUITE["JetSpace"]["rand"] = @benchmarkable rand($s)
SUITE["JetSpace"]["zeros"] = @benchmarkable zeros($s)
SUITE["JetSpace"]["ones"] = @benchmarkable ones($s)
SUITE["JetSpace"]["reshape"] = @benchmarkable reshape($x, $s)

dom = JetSpace(Float64, 20)
rng = JetSpace(Float64, 10, 2)
f!(d,m;a) = d .= a .* m.^2
df!(δd,δm;a,mₒ) = δd .= 2 .* a .* mₒ .* δm
a = rand(20)
✈ = Jet(dom=dom, rng=rng, f! = f!, df! = df!, df′! = df!, s = (a = a,))
SUITE["✈"] = BenchmarkGroup()
SUITE["✈"]["construct"] = @benchmarkable Jet(dom=$dom, rng=$rng, f! = $(f!), df! = $(df!), df′! = $(df!), s = (a = $a,))
SUITE["✈"]["domain"] = @benchmarkable domain($✈)
SUITE["✈"]["range"] = @benchmarkable range($✈)
SUITE["✈"]["point"] = @benchmarkable point($✈)
SUITE["✈"]["state"] = @benchmarkable state($✈)
SUITE["✈"]["state!"] = @benchmarkable state!($✈, (a=$(rand(20)),))
SUITE["✈"]["shape"] = @benchmarkable shape($✈)
SUITE["✈"]["size"] = @benchmarkable size($✈)

JopFoo_df!(d,m;diagonal,kwargs...) = d .= diagonal .* m
function JopFoo(diag)
    spc = JetSpace(Float64, length(diag))
    JopLn(;df! = JopFoo_df!, dom = spc, rng = spc, s = (diagonal=diag,))
end
A = JopFoo(rand(100))
m = rand(100)
d = rand(100)
SUITE["JopLn"] = BenchmarkGroup()
SUITE["JopLn"]["construct"] = @benchmarkable JopFoo($(rand(100)))
SUITE["JopLn"]["mul!"] = @benchmarkable mul!($d, $A, $m)
SUITE["JopLn"]["mul!,adjoint"] = @benchmarkable mul!($m, $(A)', $d)
SUITE["JopLn"]["mul"] = @benchmarkable $(A) * $(m)
SUITE["JopLn"]["mul, adjoint"] = @benchmarkable $(A)' * $(d)
SUITE["JopLn"]["adjoint"] = @benchmarkable $(A)'
SUITE["JopLn"]["size"] = @benchmarkable size($A)
SUITE["JopLn"]["shape"] = @benchmarkable shape($A)
SUITE["JopLn"]["domain"] = @benchmarkable domain($A)
SUITE["JopLn"]["range"] = @benchmarkable range($A)

JopBar_f!(d,m;kwargs...) = d .= m.^2
JopBar_df!(δd,δm;mₒ,kwargs...) = δd .= 2 .* mₒ .* δm
function JopBar(n)
    spc = JetSpace(Float64, n)
    JopNl(f! = JopBar_f!, df! = JopBar_df!, dom = spc, rng = spc)
end
F = JopBar(100)
m = rand(100)
d = rand(100)
SUITE["JopNl"] = BenchmarkGroup()
SUITE["JopNl"]["construct"] = @benchmarkable JopBar(100)
SUITE["JopNl"]["mul!"] = @benchmarkable mul!($d, $F, $m)
SUITE["JopNl"]["mul"] = @benchmarkable $F * $m
SUITE["JopNl"]["jacobian"] = @benchmarkable jacobian($F, $m)
SUITE["JopNl"]["size"] = @benchmarkable size($F)
SUITE["JopNl"]["shape"] = @benchmarkable shape($F)
SUITE["JopNl"]["domain"] = @benchmarkable domain($F)
SUITE["JopNl"]["range"] = @benchmarkable range($F)

G = F ∘ A ∘ F ∘ A
J = jacobian(G, m)
SUITE["Composition"] = BenchmarkGroup()
SUITE["Composition"]["construct"] = @benchmarkable $F ∘ $A ∘ $F ∘ $A
SUITE["Composition"]["mul!"] = @benchmarkable mul!($d, $G, $m)
SUITE["Composition"]["mul"] = @benchmarkable $G * $m
SUITE["Composition"]["jacobian"] = @benchmarkable jacobian($G, $m)
SUITE["Composition"]["mul!,adjoint"] = @benchmarkable mul!($m, ($J)', $d)
SUITE["Composition"]["mul, adjoint"] = @benchmarkable ($J)' * $d
SUITE["Composition"]["adjoint"] = @benchmarkable ($J)'
SUITE["Composition"]["size"] = @benchmarkable size($G)
SUITE["Composition"]["shape"] = @benchmarkable shape($G)
SUITE["Composition"]["domain"] = @benchmarkable domain($G)
SUITE["Composition"]["range"] = @benchmarkable range($G)

_F = [JopBar(100) JopBar(100) JopBar(100) ; JopBar(100) JopBar(100) JopBar(100)]
F = @blockop _F
domainF = domain(F)
m = rand(domain(F))
d = rand(range(F))
e = rand(range(F))
f = rand(range(F))
d′ = rand(Float64,size(range(F)))
e′ = rand(Float64,size(range(F)))
f′ = rand(Float64,size(range(F)))
J = jacobian(F, m)
SUITE["Block, homogeneous"] = BenchmarkGroup()
SUITE["Block, homogeneous"]["construct"] = @benchmarkable @blockop $_F
SUITE["Block, homogeneous"]["mul!"] = @benchmarkable mul!($d, $F, $m)
SUITE["Block, homogeneous"]["mul"] = @benchmarkable $F * $m
SUITE["Block, homogeneous"]["jacobian"] = @benchmarkable jacobian($F, $m)
SUITE["Block, homogeneous"]["mul!, adjoint"] = @benchmarkable mul!($m, ($J)', $d)
SUITE["Block, homogeneous"]["mul, adjoint"] = @benchmarkable ($J)' * $d
SUITE["Block, homogeneous"]["adjoint"] = @benchmarkable ($J)'
SUITE["Block, homogeneous"]["shape"] = @benchmarkable shape($F)
SUITE["Block, homogeneous"]["size"] = @benchmarkable size($F)
SUITE["Block, homogeneous"]["domain"] = @benchmarkable domain($F)
SUITE["Block, homogeneous"]["range"] = @benchmarkable range($F)
SUITE["Block, homogeneous"]["block"] = @benchmarkable getblock($m, 2)
SUITE["Block, homogeneous"]["block!"] = @benchmarkable setblock!($m, 2, $(rand(100)))
SUITE["Block, homogeneous"]["broadcast"] = @benchmarkable f .= d .+ e
SUITE["Block, homogeneous"]["broadcast (base-case)"] = @benchmarkable f′ .= d′ .+ e′

x = rand(100)
_F = [JopBar(100) JopFoo(x) JopBar(100) ; JopBar(100) JopFoo(x) JopBar(100)]
F = @blockop _F
domainF = domain(F)
m = rand(domain(F))
d = rand(range(F))
e = rand(range(F))
f = rand(range(F))
d′ = rand(Float64,size(range(F)))
e′ = rand(Float64,size(range(F)))
f′ = rand(Float64,size(range(F)))
J = jacobian(F, m)
SUITE["Block, heterogeneous"] = BenchmarkGroup()
SUITE["Block, heterogeneous"]["construct"] = @benchmarkable @blockop $_F
SUITE["Block, heterogeneous"]["mul!"] = @benchmarkable mul!($d, $F, $m)
SUITE["Block, heterogeneous"]["mul"] = @benchmarkable $F * $m
SUITE["Block, heterogeneous"]["jacobian"] = @benchmarkable jacobian($F, $m)
SUITE["Block, heterogeneous"]["mul!, adjoint"] = @benchmarkable mul!($m, ($J)', $d)
SUITE["Block, heterogeneous"]["mul, adjoint"] = @benchmarkable ($J)' * $d
SUITE["Block, heterogeneous"]["adjoint"] = @benchmarkable ($J)'
SUITE["Block, heterogeneous"]["shape"] = @benchmarkable shape($F)
SUITE["Block, heterogeneous"]["size"] = @benchmarkable size($F)
SUITE["Block, heterogeneous"]["domain"] = @benchmarkable domain($F)
SUITE["Block, heterogeneous"]["range"] = @benchmarkable range($F)
SUITE["Block, heterogeneous"]["block"] = @benchmarkable getblock($m, 2)
SUITE["Block, heterogeneous"]["block!"] = @benchmarkable setblock!($d, 2, $(rand(100)))
SUITE["Block, heterogeneous"]["broadcast"] = @benchmarkable f .= d .+ e
SUITE["Block, heterogeneous"]["broadcast (base-case)"] = @benchmarkable f′ .= d′ .+ e′

SUITE
