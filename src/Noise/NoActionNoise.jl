using SparseArrays
# TODO: this is a weak dependency

struct NoActionNoise{TA} <: AbstractActionNoise{TA}
    action::TA
end

const NoProcessNoise{TA} = NoActionNoise{TA}

get_lie_covariance_at(n::NoActionNoise, ::Any) = let d = manifold_dimension(base_group(n.action))
    return SparseArrays.spzeros(d, d)
end

get_lie_covariance_at(n::NoActionNoise, x::Any, ::Any) = get_lie_covariance_at(n, x)

add_noise(::NoActionNoise, ::Random.AbstractRNG, x) = x

rescale_noise(n::NoActionNoise, ::Any) = n

# TODO: `get_covariance_at` is missing
