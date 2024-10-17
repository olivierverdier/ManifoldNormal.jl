
"""
A model of a map from a manifold to a covariance on a Lie algebra.
"""
abstract type AbstractActionNoise{TA<:AbstractGroupAction{LeftAction}} <: AbstractNoise end

const AbstractProcessNoise{TA} = AbstractActionNoise{TA}

#--------------------------------
# AbstractActionNoise Interface
#--------------------------------
"""
    get_lie_covariance_at(
      ::AbstractActionNoise,
      x, # point in manifold M
      B, # AbstractBasis of Alg(G)
      ) :: AbstractPDMat

Covariance on Alg(G), where G is the action group.
"""
function get_lie_covariance_at end

#--------------------------------

@doc raw"""
    ActionNoise(
        action, # group action G ⊂ Diff(M)
        covariance, # function M -> AbstractPDMat
        basis # fixed basis of Alg(G)
    )

Covariance defined on Lie algebra.
The corresponding distribution centred at x₀ 
is the push forward of the normal distribution on the Lie
algebra by the function ξ ↦ \exp(ξ)⋅x₀.
"""
struct ActionNoise{TA,TF<:Function,TB} <: AbstractActionNoise{TA}
    action::TA # group action G ⊂ Diff(M)
    covariance::TF # function M -> AbstractPDMat
    basis::TB # fixed basis of Alg(G)
end

const ProcessNoise{TA} = ActionNoise{TA}

Base.show(io::IO, n::ActionNoise) = print(io, "ActionNoise($(n.action), $(n.covariance), $(n.basis))")


sample_space(a::AbstractActionNoise)  = group_manifold(a.action)

"""
    ActionNoise(A::GroupAction, Σ::PDMat)

Convenience method to create an action noise with a constant covariance Σ, by default with respect to the standard scalar product of the Lie algebra.
"""
ActionNoise(
    A::AbstractGroupAction,
    Σ::PDMats.AbstractPDMat,
    B=DefaultOrthonormalBasis()) = ActionNoise(A, Returns(Σ), B)

"""
    ActionNoise(A::GroupAction, σ::Number)

Convenience method to create an action noise with a constant, isotropic covariance, by default with respect to the standard metric of the Lie algebra.
"""
function ActionNoise(
    A::AbstractGroupAction,
    σ::Number=1.0,
    B=DefaultOrthonormalBasis())
    G = base_group(A)
    dim = manifold_dimension(G)
    return ActionNoise(A, PDMats.ScalMat(dim, σ), B)
end

rescale_noise(n::ActionNoise, scale) = ActionNoise(n.action, x -> scale^2*n.covariance(x), n.basis)

rescale_noise(n::ActionNoise{<:Any,<:Returns}, scale) = ActionNoise(n.action, Returns(scale^2*n.covariance.value), n.basis)

"""
    update_cov(n::ActionNoise, Σ)

New action noise with same action and basis but with new (possibly point dependent) covariance Σ.
"""
update_cov(n::ActionNoise, Σ) = ActionNoise(n.action, Σ, n.basis)

constant_noise_at(n::ActionNoise, x) = update_cov(n, Returns(get_lie_covariance_at(n, x)))

_basis_error_message(B1, B2) = "Changing from basis\n\t$B1\nto\n\t$B2\nis not implemented"

function get_lie_covariance_at(
    noise::ActionNoise, # noise for action(M,G)
    x, # on M
    B::AbstractBasis, # basis of Alg(G)
    ) 
    if !is_point(sample_space(noise), x)
        throw(ErrorException("x should be a point on the manifold"))
    end
    if B != noise.basis
        throw(ErrorException(_basis_error_message(B, noise.basis)))
    end
    return noise.covariance(x)
end

get_lie_covariance_at(noise::ActionNoise, x) = get_lie_covariance_at(noise, x, noise.basis)

function get_covariance_at(
    noise::ActionNoise,
    x, # point on Manifold M
    B # basis of T_xM
    ) 
    A = noise.action
    Σ = noise.covariance(x)
    BG = noise.basis
    P = GU.get_proj_matrix(A, x, BG, B)
    return PDMats.AbstractPDMat(PDMats.X_A_Xt(Σ, P))
end

function get_covariance_at(
    noise::ActionNoise{TA},
    ::Identity,
    B
    ) where {TA<:GroupOperationAction}
    if B != noise.basis
        throw(ErrorException(_basis_error_message(B, noise.basis)))
    end
    G = base_group(noise.action)
    return noise.covariance(Identity(G))
end

function add_noise(
    noise::ActionNoise,
    rng::Random.AbstractRNG,
    point)
    cov = noise.covariance(point)
    dist = ActionDistribution(noise.action, point, cov, noise.basis)
    return rand(rng, dist)
end

add_noise(noise::ActionNoise{<:Any, <:Returns}, rng::Random.AbstractRNG, point) = rand(rng, ActionDistribution(point, noise))

_side_to_direction(::LeftSide) = RightAction()
_side_to_direction(::RightSide) = LeftAction()

@doc raw"""
    adjoint_noise(noise::ActionNoise{<:GroupOperationAction})

If the noise is defined by a group operation action,
compute the corresponding noise for the action with switched sides.
It is based on the identities
```math
χ \exp(ξ) = \exp(χ ξ χ^{-1}) χ
```
and
```math
\exp(ξ) χ = χ \exp(χ^{-1} ξ χ)
```
"""
adjoint_noise(noise::AbstractActionNoise{<:GroupOperationAction{<:LeftAction, S}}) where S = begin
    G = base_group(noise.action)
    side = S()
    action_ = GroupOperationAction(G, (LeftAction(), switch_side(side)))
    return _compute_adjoint_noise(noise, G, action_, side)
end

_compute_adjoint_noise(noise::ActionNoise, G, action_, side) = begin
    cov_(χ) = begin
        ad = GU.matrix_from_lin_endomorphism(G, ξ -> adjoint_action(G, χ, ξ, _side_to_direction(side)), noise.basis)
        return PDMats.AbstractPDMat(PDMats.X_A_Xt(noise.covariance(χ), ad))
    end
    return ActionNoise(action_, cov_, noise.basis)
end


# COV_EXCL_START
"""
Obsolete: in this basis, the covariance matrix is the identity.
"""
function get_adapted_basis_at(noise::ActionNoise, x)
    A = noise.action
    BM = ManifoldsBase.get_basis(group_manifold(A), x, DefaultOrthonormalBasis())
    L = GU.get_proj_matrix(A, x, noise.basis, BM)
    res = LinearAlgebra.svd(L)
    vec_mat = res.U .* res.S'
    return CachedBasis(DefaultOrthonormalBasis(), collect(eachcol(vec_mat)))
end
# COV_EXCL_STOP
