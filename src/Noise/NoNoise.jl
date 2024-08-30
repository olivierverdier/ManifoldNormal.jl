"""
    NoNoise(M::AbstractManifold)

Models the absence of noise on the given sample space `M`.
"""
struct NoNoise{TM} <: AbstractNoise
    manifold::TM
end

Base.show(io::IO, n::NoNoise) = print(io, "NoNoise($(n.manifold))")

sample_space(n::NoNoise) = n.manifold

@doc raw"""
    get_covariance_at(n::NoNoise, ::Any, ::Any)

Returns a zero covariance matrix, as this is a deterministic distribution.
"""
get_covariance_at(n::NoNoise, ::Any, ::Any) = PDMats.ScalMat(manifold_dimension(sample_space(n)), 0.)


@doc raw"""
    add_noise(::NoNoise, ::Random.AbstractRNG, x)

The identity function. No noise is added.
"""
add_noise(::NoNoise, ::Random.AbstractRNG, x) = x

rescale_noise(n::NoNoise, ::Any) = n
