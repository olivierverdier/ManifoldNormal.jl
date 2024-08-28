module ManifoldNormal

using Manifolds
import ManifoldsBase
using ConstantFunctions

import Random
import Distributions
import PDMats
import PDMatsSingular: sample
import ManifoldGroupUtils as GU

include("Noise.jl")
include("ProjLogNormal.jl")

# Noise
export AbstractNoise, AbstractActionNoise,
    ActionNoise, IsotropicNoise,
    NoNoise,
    sample_space,
    get_covariance_at,
    update_cov

export AbstractProjLogNormal, ProjLogNormal,
    action_noise, scaled_distance,
    update_mean_cov, update_mean,
    sample

export get_action

end
