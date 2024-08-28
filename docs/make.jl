if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(Pkg.PackageSpec(; path=(@__DIR__) * "/../"))
    Pkg.resolve()
    Pkg.instantiate()
end


using Documenter
using ManifoldNormal
using ManifoldsBase
import Random


makedocs(
    sitename = "ManifoldNormal",
    format = Documenter.HTML(),
    modules = [ManifoldNormal]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/olivierverdier/ManifoldNormal.jl.git",
    push_preview=true,
)
