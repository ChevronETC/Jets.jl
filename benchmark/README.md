# Benchmarking
We use [PkgBenchmark.jl](http://github.com/juliaCI/PkgBenchmark.jl) which can be
installed using `Pkg.add("PkgBenchmark")`.  To run the benchmarks:
```julia
using PkgBenchmark
results=benchmarkpkg("Jets")
export_markdown("results.md", results)
```
In order to compare the benchmarks against a different version:
```julia
results=judge("Jets", "e1d4076")
export_markdown("results.md", results)
```
where `e1d4076` is a Git SHA or the name of a Git branch.  To run a specific
benchmark:
```julia
benchmarks=include("benchmarks.jl")
run(benchmarks["Jet"]["construct"])
```

You can profile a benchmark.  For example:
```julia
benchmarks=include("benchmarks.jl")
using Profile
@profile run(benchmarks["Jet"]["construct"])
```
Use `Profile.print()` and `using ProfileView; ProfileView.view()` to inspect the
profile.  Note that `ProfileView.view()` requires
[ProfileView.jl](http://github.com/timholy/ProfileView.jl).

For more information, please see the documentation for
[PkgBenchmark.jl](http://github.com/juliaCI/PkgBenchmark.jl) and
[BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).
