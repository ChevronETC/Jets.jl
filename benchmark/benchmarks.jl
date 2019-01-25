using BenchmarkTools, JotNew, LinearAlgebra

const SUITE = BenchmarkGroup()

s = JotSpace(Float64, 100, 200)
x = rand(20_000)
SUITE["JotSpace"] = BenchmarkGroup()
SUITE["JotSpace"]["construct"] = @benchmarkable JotSpace(Float64, 2, 3)
SUITE["JotSpace"]["size"] = @benchmarkable size($s)
SUITE["JotSpace"]["eltype"] = @benchmarkable eltype($s)
SUITE["JotSpace"]["ndims"] = @benchmarkable ndims($s)
SUITE["JotSpace"]["rand"] = @benchmarkable rand($s)
SUITE["JotSpace"]["zeros"] = @benchmarkable zeros($s)
SUITE["JotSpace"]["ones"] = @benchmarkable ones($s)
SUITE["JotSpace"]["reshape"] = @benchmarkable reshape($x, $s)

dom = JotSpace(Float64, 20)
rng = JotSpace(Float64, 10, 2)
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

function JotOpFoo(diag)
    df!(d,m;diagonal,kwargs...) = d .= diagonal .* m
    spc = JotSpace(Float64, length(diag))
    JotOpLn(;df! = df!, df′! = df!, dom = spc, rng = spc, s = (diagonal=diag,))
end
A = JotOpFoo(rand(100))
m = rand(100)
d = rand(100)
SUITE["JotOpLn"] = BenchmarkGroup()
SUITE["JotOpLn"]["construct"] = @benchmarkable JotOpFoo($(rand(100)))
SUITE["JotOpLn"]["mul!"] = @benchmarkable mul!($d, $A, $m)
SUITE["JotOpLn"]["mul!,adjoint"] = @benchmarkable mul!($m, $(A)', $d)
SUITE["JotOpLn"]["mul"] = @benchmarkable $(A) * $(m)
SUITE["JotOpLn"]["mul, adjoint"] = @benchmarkable $(A)' * $(d)
SUITE["JotOpLn"]["adjoint"] = @benchmarkable $(A)'
SUITE["JotOpLn"]["size"] = @benchmarkable size($A)
SUITE["JotOpLn"]["shape"] = @benchmarkable shape($A)
SUITE["JotOpLn"]["domain"] = @benchmarkable domain($A)
SUITE["JotOpLn"]["range"] = @benchmarkable range($A)

function JotOpBar(n)
    f!(d,m) = d .= m.^2
    df!(δd,δm;mₒ,kwargs...) = δd .= 2 .* mₒ .* δm
    spc = JotSpace(Float64, n)
    JotOpNl(f! = f!, df! = df!, df′! = df!, dom = spc, rng = spc)
end
F = JotOpBar(100)
m = rand(100)
d = rand(100)
SUITE["JotOpNl"] = BenchmarkGroup()
SUITE["JotOpNl"]["construct"] = @benchmarkable JotOpBar(100)
SUITE["JotOpNl"]["mul!"] = @benchmarkable mul!($d, $F, $m)
SUITE["JotOpNl"]["mul"] = @benchmarkable $F * $m
SUITE["JotOpNl"]["jacobian"] = @benchmarkable jacobian($F, $m)
SUITE["JotOpNl"]["size"] = @benchmarkable size($F)
SUITE["JotOpNl"]["shape"] = @benchmarkable shape($F)
SUITE["JotOpNl"]["domain"] = @benchmarkable domain($F)
SUITE["JotOpNl"]["range"] = @benchmarkable range($F)

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

_F = [JotOpBar(100) JotOpBar(100) JotOpBar(100) ; JotOpBar(100) JotOpBar(100) JotOpBar(100)]
F = @blockop _F
m = rand(domain(F))
d = rand(range(F))
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
SUITE["Block, homogeneous"]["getblockdomain"] = @benchmarkable getblockdomain($m, $F, 2)
SUITE["Block, homogeneous"]["getblockrange"] = @benchmarkable getblockrange($d, $F, 2)
SUITE["Block, homogeneous"]["setblockdomain!"] = @benchmarkable setblockdomain!($m, $F, 2, $(rand(100)))
SUITE["Block, homogeneous"]["setblockrange!"] = @benchmarkable setblockrange!($d, $F, 2, $(rand(100)))

x = rand(100)
_F = [JotOpBar(100) JotOpFoo(x) JotOpBar(100) ; JotOpBar(100) JotOpFoo(x) JotOpBar(100)]
F = @blockop _F
m = rand(domain(F))
d = rand(range(F))
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
SUITE["Block, homogeneous"]["getblockdomain"] = @benchmarkable getblockdomain($m, $F, 2)
SUITE["Block, homogeneous"]["getblockrange"] = @benchmarkable getblockrange($d, $F, 2)
SUITE["Block, homogeneous"]["setblockdomain!"] = @benchmarkable setblockdomain!($m, $F, 2, $(rand(100)))
SUITE["Block, homogeneous"]["setblockdomain"] = @benchmarkable setblockrange!($d, $F, 2, $(rand(100)))

SUITE
