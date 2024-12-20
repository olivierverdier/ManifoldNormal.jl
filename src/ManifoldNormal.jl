module ManifoldNormal

using Manifolds
import ManifoldsBase

import Random
import Distributions
import PDMats
import PDMatsSingular: sample
import ManifoldGroupUtils as GU

include("Noise.jl")
include("ActionDistribution.jl")

# Noise
export AbstractNoise,
    AbstractActionNoise, AbstractProcessNoise,
    ActionNoise, ProcessNoise,
    NoActionNoise, NoProcessNoise,
    IsotropicNoise,
    NoNoise,
    sample_space,
    get_covariance_at,
    update_cov,
    adjoint_noise, adjoint_distribution

export AbstractActionDistribution, AbstractProcessDistribution,
    ActionDistribution, ProcessDistribution,
    action_noise, scaled_distance,
    update_mean_cov, update_mean,
    sample

# deprecated
const AbstractProjLogNormal = AbstractActionDistribution
@deprecate ProjLogNormal(args...) ActionDistribution(args...)
export ProjLogNormal, AbstractProjLogNormal

export get_action

end
