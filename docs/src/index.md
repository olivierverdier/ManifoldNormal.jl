# ManifoldNormal.jl

## Noise Models

A general interface for noises on manifolds.

```jldoctest noise_example
using ManifoldsBase
using ManifoldNormal
σ = 2.0 # the standard deviation
M = ManifoldsBase.DefaultManifold(2)
# choose an isotropic noise for this example
noise = IsotropicNoise(M, σ)
x = rand(M)
""
# output
""
```

```jldoctest noise_example
sample_space(noise)
# output
DefaultManifold(2; field = ℝ)
```

```jldoctest noise_example
Σ = get_covariance_at(noise, x, DefaultOrthonormalBasis()) 
# output
2×2 PDMats.ScalMat{Float64}:
 4.0  0.0
 0.0  4.0
```

```jldoctest noise_example
import Random
rng = Random.default_rng()
y = noise(rng, x) # another point of M
using Manifolds
is_point(M, y)
# output
true
```

```@autodocs
Modules = [ManifoldNormal]
Pages = ["Noise.jl", "Noise/Action.jl", "Noise/Isotropic.jl", "Noise/NoNoise.jl"]
Order = [:type, :function]
```

## Normal Distributions


```@autodocs
Modules = [ManifoldNormal]
Pages = ["ProjLogNormal.jl"]
Order = [:type, :function]
```
