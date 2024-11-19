var documenterSearchIndex = {"docs":
[{"location":"#ManifoldNormal.jl","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"","category":"section"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"This packages defines distributions and noises on manifolds. They are defined as follows:","category":"page"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"A distribution is a fully prescribed probability distribution on a given manifold; an example is the ActionDistribution distribution.\nA noise is a distribution on a manifold, but parameterized by a point on that manifold; examples are IsotropicNoise, ActionNoise and NoNoise.","category":"page"},{"location":"#Noise-Models","page":"ManifoldNormal.jl","title":"Noise Models","text":"","category":"section"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"A general interface for noises on manifolds.","category":"page"},{"location":"#Example-Usage","page":"ManifoldNormal.jl","title":"Example Usage","text":"","category":"section"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"Let us set up a simple manifold and a point on it.","category":"page"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"using ManifoldsBase\nusing ManifoldNormal\nσ = 2.0 # the standard deviation\nM = ManifoldsBase.DefaultManifold(2)\n# choose an isotropic noise for this example\nnoise = IsotropicNoise(M, σ)\nx = rand(M)\n\"\"\n# output\n\"\"","category":"page"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"The sample space is the space where noisy points are expected.","category":"page"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"sample_space(noise)\n# output\nDefaultManifold(2; field = ℝ)","category":"page"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"The covariance of the noise at the point x. It is a covariance operator on the tangent space at x.","category":"page"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"Σ = get_covariance_at(noise, x, DefaultOrthonormalBasis()) \n# output\n2×2 PDMats.ScalMat{Float64}:\n 4.0  0.0\n 0.0  4.0","category":"page"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"Using noise(rng, x) to produce sample from the noise distribution centred at x.","category":"page"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"import Random\nrng = Random.default_rng()\ny = noise(rng, x) # another point of M\nusing Manifolds\nis_point(M, y)\n# output\ntrue","category":"page"},{"location":"#Interface","page":"ManifoldNormal.jl","title":"Interface","text":"","category":"section"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"tip: AbstractNoise interface\nAll the noises are subtypes of the AbstractNoise abstract type. The following methods are expected:sample_space\nget_covariance_at\nManifoldNormal.get_basis_at\nManifoldNormal.add_noise\nManifoldNormal.rescale_noise","category":"page"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"","category":"page"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"Modules = [ManifoldNormal]\nPages = [\"Noise.jl\"]\nOrder = [:function]","category":"page"},{"location":"#ManifoldNormal.add_noise","page":"ManifoldNormal.jl","title":"ManifoldNormal.add_noise","text":"add_noise(::AbstractNoise, ::Random.AbstractRNG, x::TX) :: TX\n\nAdd noise at point x.\n\nShortcut: noise(rng, x) ≡ add_noise(noise, rng, x)\n\n\n\n\n\n","category":"function"},{"location":"#ManifoldNormal.add_noise-Tuple{NoNoise, Random.AbstractRNG, Any}","page":"ManifoldNormal.jl","title":"ManifoldNormal.add_noise","text":"add_noise(::NoNoise, ::Random.AbstractRNG, x)\n\nThe identity function. No noise is added.\n\n\n\n\n\n","category":"method"},{"location":"#ManifoldNormal.get_basis_at","page":"ManifoldNormal.jl","title":"ManifoldNormal.get_basis_at","text":"get_basis_at(::AbstractNoise, x) :: AbstractBasis\n\nThe basis given by the noise model at the point x.\n\n\n\n\n\n","category":"function"},{"location":"#ManifoldNormal.get_covariance_at","page":"ManifoldNormal.jl","title":"ManifoldNormal.get_covariance_at","text":"get_covariance_at(::AbstractNoise, x, ::AbstractBasis)  :: AbstractPDMat\n\nA covariance matrix for this noise, at the given point,  expressed in the given basis.\n\n\n\n\n\n","category":"function"},{"location":"#ManifoldNormal.get_covariance_at-Tuple{NoNoise, Any, Any}","page":"ManifoldNormal.jl","title":"ManifoldNormal.get_covariance_at","text":"get_covariance_at(n::NoNoise, ::Any, ::Any)\n\nReturns a zero covariance matrix, as this is a deterministic distribution.\n\n\n\n\n\n","category":"method"},{"location":"#ManifoldNormal.rescale_noise","page":"ManifoldNormal.jl","title":"ManifoldNormal.rescale_noise","text":"rescale_noise(TN<:AbstractNoise, λ) :: TN\n\nRescale the noise by a given factor λ. If X is the random variable distributed as the noise, then return the distribution of λX.\n\nShortcut: λ * noise ≡ rescale_noise(noise, λ)\n\n\n\n\n\n","category":"function"},{"location":"#ManifoldNormal.sample_space","page":"ManifoldNormal.jl","title":"ManifoldNormal.sample_space","text":"sample_space(::AbstractNoise) :: AbstractManifold\n\nManifold on which the noise operates.\n\n\n\n\n\n","category":"function"},{"location":"#Methods","page":"ManifoldNormal.jl","title":"Methods","text":"","category":"section"},{"location":"#Action-Noise","page":"ManifoldNormal.jl","title":"Action Noise","text":"","category":"section"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"Consider a group action G  mathrmDiff(M). The push-forward of a normal distribution on a Lie algebra. The push-forward at a point xin M is done by the function F  mathfrakg to M defined by","category":"page"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"F(ξ) = exp(ξ) x","category":"page"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"Modules = [ManifoldNormal]\nPages = [\"Noise/Action.jl\"]\nOrder = [:type, :function]","category":"page"},{"location":"#ManifoldNormal.AbstractActionNoise","page":"ManifoldNormal.jl","title":"ManifoldNormal.AbstractActionNoise","text":"A model of a map from a manifold to a covariance on a Lie algebra.\n\n\n\n\n\n","category":"type"},{"location":"#ManifoldNormal.ActionNoise","page":"ManifoldNormal.jl","title":"ManifoldNormal.ActionNoise","text":"ActionNoise(A::GroupAction, Σ::PDMat)\n\nConvenience method to create an action noise with a constant covariance Σ, by default with respect to the standard scalar product of the Lie algebra.\n\n\n\n\n\n","category":"type"},{"location":"#ManifoldNormal.ActionNoise-2","page":"ManifoldNormal.jl","title":"ManifoldNormal.ActionNoise","text":"ActionNoise(A::GroupAction, σ::Number)\n\nConvenience method to create an action noise with a constant, isotropic covariance, by default with respect to the standard metric of the Lie algebra.\n\n\n\n\n\n","category":"type"},{"location":"#ManifoldNormal.ActionNoise-3","page":"ManifoldNormal.jl","title":"ManifoldNormal.ActionNoise","text":"ActionNoise(\n    action, # group action G ⊂ Diff(M)\n    covariance, # function M -> AbstractPDMat\n    basis # fixed basis of Alg(G)\n)\n\nCovariance defined on Lie algebra. The corresponding distribution centred at x₀  is the push forward of the normal distribution on the Lie algebra by the function ξ ↦ \\exp(ξ)⋅x₀.\n\n\n\n\n\n","category":"type"},{"location":"#ManifoldNormal.adjoint_noise-Union{Tuple{AbstractProcessNoise{<:Manifolds.GroupOperationAction{<:Manifolds.LeftAction, S}}}, Tuple{S}} where S","page":"ManifoldNormal.jl","title":"ManifoldNormal.adjoint_noise","text":"adjoint_noise(noise::ActionNoise{<:GroupOperationAction})\n\nIf the noise is defined by a group operation action, compute the corresponding noise for the action with switched sides. It is based on the identities\n\nχ exp(ξ) = exp(χ ξ χ^-1) χ\n\nand\n\nexp(ξ) χ = χ exp(χ^-1 ξ χ)\n\n\n\n\n\n","category":"method"},{"location":"#ManifoldNormal.get_adapted_basis_at-Tuple{ActionNoise, Any}","page":"ManifoldNormal.jl","title":"ManifoldNormal.get_adapted_basis_at","text":"Obsolete: in this basis, the covariance matrix is the identity.\n\n\n\n\n\n","category":"method"},{"location":"#ManifoldNormal.get_lie_covariance_at","page":"ManifoldNormal.jl","title":"ManifoldNormal.get_lie_covariance_at","text":"get_lie_covariance_at(\n  ::AbstractActionNoise,\n  x, # point in manifold M\n  B, # AbstractBasis of Alg(G)\n  ) :: AbstractPDMat\n\nCovariance on Alg(G), where G is the action group.\n\n\n\n\n\n","category":"function"},{"location":"#ManifoldNormal.update_cov-Tuple{ActionNoise, Any}","page":"ManifoldNormal.jl","title":"ManifoldNormal.update_cov","text":"update_cov(n::ActionNoise, Σ)\n\nNew action noise with same action and basis but with new (possibly point dependent) covariance Σ.\n\n\n\n\n\n","category":"method"},{"location":"#Isotropic-Noise","page":"ManifoldNormal.jl","title":"Isotropic Noise","text":"","category":"section"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"This noise at a point x on a metric manifold M is the push-forward by the metric exponential of a centred normal distribution on the tangent space T_x M. The covariance of that normal distribution is equal to the metric at that point x.","category":"page"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"Modules = [ManifoldNormal]\nPages = [\"Noise/Isotropic.jl\"]\nOrder = [:type, :function]","category":"page"},{"location":"#ManifoldNormal.IsotropicNoise","page":"ManifoldNormal.jl","title":"ManifoldNormal.IsotropicNoise","text":"IsotropicNoise(manifold, # metric manifold M\n  deviation # function M -> 𝑹\n)\n\nModel a noise with covariance equal to the metric at the given point on the manifold.\n\n\n\n\n\n","category":"type"},{"location":"#ManifoldNormal.IsotropicNoise-Tuple{Any, Number}","page":"ManifoldNormal.jl","title":"ManifoldNormal.IsotropicNoise","text":"IsotropicNoise(M, std::Number)\n\nCreate an isotropic noise on the manifold M. The number std is then the standard deviation, which does not depend on the point of the manifold.\n\n\n\n\n\n","category":"method"},{"location":"#ManifoldNormal.isotropic_perturbation-Tuple{Random.AbstractRNG, Any, Any}","page":"ManifoldNormal.jl","title":"ManifoldNormal.isotropic_perturbation","text":"isotropic_perturbation(rng, M, p)\n\nCreate noisy vector at the point p on the metric manifold M, where the covariance matrix is the identity in an orthonormal basis.\n\n\n\n\n\n","category":"method"},{"location":"#No-Noise","page":"ManifoldNormal.jl","title":"No Noise","text":"","category":"section"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"A convenient noise which is just the point-mass distribution at a point x on any manifold.","category":"page"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"Modules = [ManifoldNormal]\nPages = [\"Noise/NoNoise.jl\"]\nOrder = [:type]","category":"page"},{"location":"#ManifoldNormal.NoNoise","page":"ManifoldNormal.jl","title":"ManifoldNormal.NoNoise","text":"NoNoise(M::AbstractManifold)\n\nModels the absence of noise on the given sample space M.\n\n\n\n\n\n","category":"type"},{"location":"#Action-Distributions","page":"ManifoldNormal.jl","title":"Action Distributions","text":"","category":"section"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"This is essentially an ActionNoise but with a fixed reference point.","category":"page"},{"location":"","page":"ManifoldNormal.jl","title":"ManifoldNormal.jl","text":"Modules = [ManifoldNormal]\nPages = [\"ActionDistribution.jl\"]\nOrder = [:type, :function]","category":"page"},{"location":"#ManifoldNormal.ActionDistribution","page":"ManifoldNormal.jl","title":"ManifoldNormal.ActionDistribution","text":"ActionDistribution(μ, noise::ActionNoise)\n\nDistribution stemming from an action noise specialized at one point.\n\n\n\n\n\n","category":"type"},{"location":"#ManifoldNormal.ActionDistribution-2","page":"ManifoldNormal.jl","title":"ManifoldNormal.ActionDistribution","text":"ActionDistribution(action::Action, μ, Σ::PDMat, B::Basis)\n\nCreate an action distribution from the given action, mean and covariance.\n\n\n\n\n\n","category":"type"},{"location":"#ManifoldNormal.action_noise-Tuple{AbstractActionDistribution}","page":"ManifoldNormal.jl","title":"ManifoldNormal.action_noise","text":"action_noise(D::ActionDistribution) :: ActionNoise\n\nCreate an action noise object from a ActionDistribution distribution. This is simply an action noise with constant covariance.\n\n\n\n\n\n","category":"method"},{"location":"#ManifoldNormal.get_action","page":"ManifoldNormal.jl","title":"ManifoldNormal.get_action","text":"get_action(d::AbstractActionDistribution) :: AbstractGroupAction\n\nThe underlying action (i.e., the underlying homogeneous space).\n\n\n\n\n\n","category":"function"},{"location":"#ManifoldNormal.get_lie_basis","page":"ManifoldNormal.jl","title":"ManifoldNormal.get_lie_basis","text":"get_lie_basis(d::AbstractActionDistribution) :: Basis\n\nThe basis of mathfrakg in which the covariance is defined.\n\n\n\n\n\n","category":"function"},{"location":"#ManifoldNormal.random_vector_lie-Tuple{Random.AbstractRNG, AbstractActionDistribution}","page":"ManifoldNormal.jl","title":"ManifoldNormal.random_vector_lie","text":"random_vector_lie(\n    rng::Random.AbstractRNG,\n    d::AbstractActionDistribution) = begin\n\nDraw a random vector in the Lie algebra following the covariance in the Lie algebra.\n\n\n\n\n\n","category":"method"},{"location":"#ManifoldNormal.scaled_distance-Tuple{AbstractActionDistribution, Any}","page":"ManifoldNormal.jl","title":"ManifoldNormal.scaled_distance","text":"scaled_distance(D::ActionDistribution, x)\n\nStart with a ActionDistribution(μ,Σ) distribution with mean μ lying on the sample manifold M, and covariance Σ in the Lie algebra mathfrakg of the group G. The infinitesimal action of the group G on the manifold M gives rise to the linear map P colon mathfrakg to T_μM, which in turns gives the projected covariance PΣP^* on the tangent space T_μM.\n\nOne then measures the scaled distance from μ to x from v = log(μ x) with the formula\n\nfracsqrtv^T (PΣP^*)^-1 vsqrtn\n\n(where v is regarded as a column matrix here) and n is the dimension of the sample manifold M.\n\n\n\n\n\n","category":"method"},{"location":"#ManifoldNormal.update_mean-Tuple{ActionDistribution, Any}","page":"ManifoldNormal.jl","title":"ManifoldNormal.update_mean","text":"update_mean_cov(d::ActionDistribution, μ)\n\nReturn new ActionDistribution object with new mean μ.\n\n\n\n\n\n","category":"method"},{"location":"#ManifoldNormal.update_mean_cov-Union{Tuple{TM}, Tuple{ProcessDistribution{<:Any, TM}, TM, Any}} where TM","page":"ManifoldNormal.jl","title":"ManifoldNormal.update_mean_cov","text":"update_mean_cov(d::ActionDistribution, μ, Σ)\n\nReturn new ActionDistribution object with new mean μ and covariance Σ.\n\n\n\n\n\n","category":"method"}]
}
