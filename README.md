# ManifoldNormal

[![Build Status](https://github.com/olivierverdier/ManifoldNormal.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/olivierverdier/ManifoldNormal.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/olivierverdier/ManifoldNormal.jl/graph/badge.svg?token=SPnqp1HGCm)](https://codecov.io/gh/olivierverdier/ManifoldNormal.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://olivierverdier.github.io/ManifoldNormal.jl/)


Various instances of normal distributions on manifolds.
Here is how to use isotropic noise on a sphere.

```julia
using Manifolds
M = Sphere(2)
using ManifoldNormal
δ = .1 # standard deviation
noise = IsotropicNoise(M, δ)
import Random
rng = Random.default_rng()
x = [1., 0, 0]
noise(rng, x) # another point on the sphere, close to x
```
