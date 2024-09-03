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
    IsotropicNoise,
    NoNoise,
    sample_space,
    get_covariance_at,
    update_cov

export AbstractActionDistribution, AbstractProcessDistribution,
    ActionDistribution, ProcessDistribution,
    action_noise, scaled_distance,
    update_mean_cov, update_mean,
    sample

export get_action

end
