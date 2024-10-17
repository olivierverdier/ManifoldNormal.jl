abstract type ManifoldVariate{TM} <: Distributions.VariateForm end

abstract type AbstractActionDistribution{TA<:AbstractGroupAction{LeftAction}} <: Distributions.Sampleable{ManifoldVariate{AbstractManifold}, Distributions.Continuous} end

const AbstractProcessDistribution{TA} = AbstractActionDistribution{TA}

#--------------------------------
# AbstractActionDistribution Interface
#--------------------------------
"""
    get_action(d::AbstractActionDistribution) :: AbstractGroupAction

The underlying action (i.e., the underlying homogeneous space).
"""
function get_action end

"""
    Distribution.mean(d::AbstractActionDistribution) :: [element of manifold M]
    Distribution.cov(d::AbstractActionDistribution) :: [covariance in Alg(G)]

The mean and covariance of the distribution.
"""

@doc raw"""
    get_lie_basis(d::AbstractActionDistribution) :: Basis

The basis of ``\mathfrak{g}`` in which the covariance is defined.
"""
function get_lie_basis end

"""
    ActionDistribution(μ, noise::ActionNoise)

Distribution stemming from an action noise specialized at one point.
"""
struct ActionDistribution{TA<:AbstractGroupAction{LeftAction},TM,TN} <: AbstractActionDistribution{TA}
    μ::TM # mean: element of M
    noise::TN
    function ActionDistribution(μ::TM, noise::TN) where {TM,TN}
        @assert is_point(sample_space(noise), μ)
        return new{typeof(noise.action), TM, TN}(μ, noise)
    end
end


const ProcessDistribution{TA} = ActionDistribution{TA}

Base.show(io::IO, dist::ActionDistribution) = print(io, "ActionDistribution($(dist.μ), $(dist.noise))")

"""
    ActionDistribution(action::Action, μ, Σ::PDMat, B::Basis)

Create an action distribution from the given action, mean and covariance.
"""
ActionDistribution(action, μ, Σ, B=DefaultOrthonormalBasis()) = ActionDistribution(μ, ActionNoise(action, PDMats.AbstractPDMat(Σ), B))

function ActionDistribution(
    A, # action
    x, # in M
    σ :: Number, # isotropic variance
    B=DefaultOrthonormalBasis() :: AbstractBasis
    )
    G = base_group(A)
    dim = manifold_dimension(G)
    Σ = PDMats.ScalMat(dim, σ)
    return ActionDistribution(A, x, Σ, B)
end


Distributions.cov(d::ActionDistribution) = d.noise.covariance(d.μ)
Distributions.mean(d::ActionDistribution) = d.μ
get_action(d::ActionDistribution) = d.noise.action
get_lie_basis(d::ActionDistribution) = d.noise.basis

"""
    update_mean_cov(d::ActionDistribution, μ, Σ)

Return new `ActionDistribution` object with new mean ``μ`` and covariance ``Σ``.
"""
update_mean_cov(d::ActionDistribution{<:Any,TM}, μ::TM, Σ) where {TM}  = ActionDistribution(μ, update_cov(d.noise, Σ))

"""
    update_mean_cov(d::ActionDistribution, μ)

Return new `ActionDistribution` object with new mean ``μ``.
"""
update_mean(d::ActionDistribution, x) = update_mean_cov(d, x, Distributions.cov(d))

function Base.length(d::ActionDistribution)
    M = group_manifold(get_action(d))
    return manifold_dimension(M)
end

"""
    random_vector_lie(
        rng::Random.AbstractRNG,
        d::AbstractActionDistribution) = begin

Draw a random vector in the Lie algebra following the
covariance in the Lie algebra.
"""
random_vector_lie(
      rng::Random.AbstractRNG,
      d::AbstractActionDistribution) = begin
    rc = sample(rng, Distributions.cov(d))
    G = base_group(get_action(d))
    ξ = get_vector_lie(G, rc, get_lie_basis(d))
    return ξ
end

function rand!(
    rng::Random.AbstractRNG,
    d::AbstractActionDistribution,
    out::AbstractArray,
)
    ξ = random_vector_lie(rng, d)
    A = get_action(d)
    χ = exp_lie(base_group(A), ξ)
    apply!(A, out, χ, Distributions.mean(d))
    return out
end

function Base.rand(
    rng::Random.AbstractRNG,
    d::AbstractActionDistribution,
    )
    M = sample_space(d.noise)
    x = allocate_result(M, typeof(rand))
    rand!(rng, d, x)
    return x
end

"""
    action_noise(D::ActionDistribution) :: ActionNoise

Create an action noise object from a `ActionDistribution` distribution.
This is simply an action noise with constant covariance.
"""
action_noise(D::AbstractActionDistribution) = D.noise


@doc raw"""
    scaled_distance(D::ActionDistribution, x)

Start with a `ActionDistribution(μ,Σ)` distribution with mean ``μ`` lying
on the sample manifold ``M``, and
covariance ``Σ`` in the Lie algebra ``\mathfrak{g}`` of the group ``G``.
The infinitesimal action of the group ``G``
on the manifold ``M`` gives rise to the linear map ``P \colon \mathfrak{g} \to T_{μ}M``,
which in turns gives the projected covariance ``PΣP^*``
on the tangent space ``T_{μ}M``.

One then measures the scaled distance from
``μ`` to ``x`` from ``v := \log(μ, x)`` with the formula
```math
\frac{\sqrt{v^T (PΣP^*)^{-1} v}}{\sqrt{n}}
```
(where ``v`` is regarded as a column matrix here)
and ``n`` is the dimension of the sample manifold ``M``.
"""
function scaled_distance(D::AbstractActionDistribution, x)
    noise = action_noise(D)
    B = DefaultOrthonormalBasis()
    x0 = Distributions.mean(D)
    mat = get_covariance_at(noise, x0, B)
    M = sample_space(noise)
    vel = log(M, x0, x)
    vc = get_coordinates(M, x0, vel, B)
    vc_ = reshape(vc, :, 1)
    return sqrt(first(PDMats.Xt_A_X(mat, vc_))/manifold_dimension(M))
end

