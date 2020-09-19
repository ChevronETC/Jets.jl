using Documenter, Jets, LinearAlgebra, Random

makedocs(
    sitename="Jets.jl",
    modules=[Jets],
    pages = [ "index.md", "reference.md" ]
)

deploydocs(
    repo = "github.com/ChevronETC/Jets.jl.git"
)
