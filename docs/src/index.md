# ManifoldNormal.jl

This packages defines *distributions* and *noises* on manifolds.
They are defined as follows:
- A *distribution* is a fully prescribed probability distribution on a given manifold; an example is the [`ActionDistribution`](@ref) distribution.
- A *noise* is a distribution parameterized by a point on a manifold; examples are [`IsotropicNoise`](@ref), [`ActionNoise`](@ref) and [`NoNoise`](@ref).

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

### Interface

All the noises are subtypes of the `AbstractNoise` abstract type.
The following methods are expected:

```@autodocs
Modules = [ManifoldNormal]
Pages = ["Noise.jl"]
Order = [:function]
```



### Methods

#### Action Noise

Consider a group action ``G ⊂ \mathrm{Diff}(M)``.
The push-forward of a normal distribution on a Lie algebra.
The push-forward at a point ``x\in M`` is done by the function ``F(ξ)`` defined by
```math
F(ξ) = \exp(ξ)⋅ x
```

```@autodocs
Modules = [ManifoldNormal]
Pages = ["Noise/Action.jl"]
Order = [:type, :function]
```

#### Isotropic Noise

This noise at a point ``x`` on a metric manifold ``M``
is the push-forward by the metric exponential of a centred normal distribution
on the tangent space ``T_x M``.
The covariance of that normal distribution is equal to the metric at that point ``x``.

```@autodocs
Modules = [ManifoldNormal]
Pages = ["Noise/Isotropic.jl"]
Order = [:type, :function]
```

#### No Noise

A convenient noise which is just the point-mass distribution at a point ``x``
on any manifold.

```@autodocs
Modules = [ManifoldNormal]
Pages = ["Noise/NoNoise.jl"]
Order = [:type]
```

## Action Distributions

This is essentially an [`ActionNoise`](@ref) but with a fixed reference point.

```@autodocs
Modules = [ManifoldNormal]
Pages = ["ActionDistribution.jl"]
Order = [:type, :function]
```
