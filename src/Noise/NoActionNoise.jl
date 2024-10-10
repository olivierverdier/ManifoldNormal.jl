using SparseArrays
# TODO: this is a weak dependency

struct NoActionNoise{TA} <: AbstractActionNoise{TA}
    action::TA
end

const NoProcessNoise{TA} = NoActionNoise{TA}

get_lie_covariance_at(n::NoActionNoise, ::Any, ::Any) = let d = manifold_dimension(base_group(n.action))
    return SparseArrays.spzeros(d, d)
end

# TODO: this implementation is incomplete
