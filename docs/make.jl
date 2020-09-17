push!(LOAD_PATH,"../src/")

using Documenter, Jets

makedocs(
    sitename="Jets.jl",
    modules=[CvxCompress],
    pages = [
        "Home" => "index.md",
        "User Guide" => "manual.md",
        "reference.md"
        ]
)

deploydocs(
    repo = "github.com/ChevronETC/Jets.jl"
)
