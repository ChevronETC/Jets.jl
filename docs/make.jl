using Documenter, Jets, LinearAlgebra, Random

makedocs(sitename="Jets.jl", modules=[Jets])

deploydocs(
    repo = "github.com/ChevronETC/Jets.jl"
)
