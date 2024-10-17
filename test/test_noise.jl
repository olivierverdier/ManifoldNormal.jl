using Manifolds: LinearAlgebra
using Manifolds

using PDMats
import Random
rng = Random.default_rng()



@testset "Basic" begin
    dim = 3
    M = Euclidean(dim)
    G = TranslationGroup(dim)
    A = TranslationAction(M, G)
    δ = 2.
    noises = [
        NoNoise(M),
        IsotropicNoise(M, δ),
        IsotropicNoise(M, x -> δ),
        ActionNoise(A, δ^2),
        ActionNoise(A, x -> PDMats.ScalMat(dim, δ^2), DefaultOrthogonalBasis()),
    ]
    @testset "$noise" for noise in noises
        M = sample_space(noise)
        x = rand(rng, M)
        cov = get_covariance_at(noise, x, DefaultOrthonormalBasis())
        if noise isa NoNoise
            expected = PDMats.ScalMat(dim, 0.)
        else
            expected = PDMats.ScalMat(dim, δ^2)
        end
        @test isapprox(cov, expected)
        x_ = noise(rng, x)
        scaled = 0.0 * noise
        x__ = scaled(rng, x)
        @test isapprox(M, x, x__)
    end
end

@testset "Test Noise" begin
    G = Orthogonal(3)
    A = GroupOperationAction(G)
    B = DefaultOrthonormalBasis()
    σ = 4.0
    N = ActionNoise(A, PDMats.ScalMat(manifold_dimension(G), σ))
    N = ActionNoise(A, PDMats.ScalMat(manifold_dimension(G), σ), B)
    N_ = ActionNoise(A, σ)
    N__ = sqrt(σ) * ActionNoise(A)
    @test N_ == N
    @test N__ == N
    # get_basis_at(N, Identity(G))
    covmat = get_covariance_at(N, Identity(G))
    sn = 0.5 * N
    @test get_covariance_at(sn, Identity(G), B) ≈ covmat / 4
end

@testset "ActionNoise Basis" begin
    G = SpecialOrthogonal(3)
    M = Sphere(2)
    A = RotationAction(M, G)
    x = [1., 0, 0]
    noise = ActionNoise(A, 1.)
    B = ManifoldNormal.get_basis_at(noise, x)
    @test get_vector(M, x, [1., 0], B) ≈ [0,1,0]
end

@testset "NoNoise" begin
    dim = 2
    M = Sphere(dim)
    n = NoNoise(M)
    x = rand(M)
    rng = Random.default_rng()
    @test x == n(rng, x)
    @test n == 2.0*n
    @test get_covariance_at(n, x) == zeros(dim,dim)
end

@testset "Flat Action Noise" begin
    lin = [0 1.0; 0 0]
    trans = zeros(2)
    # motion = FlatAffineMotion(lin, trans)
    # A = AffineMotions.get_action(motion)

    x0 = [0.0, 20]
    # observer = LinearObserver([1.0 0])
    onoise = ActionNoise(TranslationAction(Euclidean(1), TranslationGroup(1)), 10.0)
    @test get_covariance_at(onoise, 0, DefaultOrthonormalBasis()) isa PDMats.AbstractPDMat
end

@testset "Rescale" begin
    G = Orthogonal(3)
    A = GroupOperationAction(G)
    B = DefaultOrthonormalBasis()
    in = IsotropicNoise(G, x->1.)
    in_ = IsotropicNoise(G, 1.0)
    @test get_covariance_at(2*in, Identity(G), B) == 4*LinearAlgebra.diagm(ones(manifold_dimension(G)))
    @test_throws MethodError get_covariance_at(in, Identity(G), DefaultOrthogonalBasis())
    an = ActionNoise(A, PDMats.ScalMat(3, 1.), B)
    an_ = ActionNoise(A, PDMats.ScalMat(3, 1.), B)
    @test get_covariance_at(2*an, Identity(G), B) == 4*LinearAlgebra.diagm(ones(manifold_dimension(G)))
    @test 2*update_cov(an, PDMats.ScalMat(3,1/4)) == an
    @test_throws ErrorException get_covariance_at(an, Identity(G), DefaultOrthogonalBasis())
end


@testset "Test Action Noise" begin
    G = SpecialOrthogonal(3)
    S = Sphere(2)
    A = RotationAction(S, G)
    x = [1., 0, 0]
    BG = DefaultOrthogonalBasis()
    BM = DefaultOrthonormalBasis()
    # noise = ActionNoise(A, x->Matrix{Float64}(LinearAlgebra.I, 3, 3), BG)
    σ = 4.0
    noise = ActionNoise(A, PDMats.ScalMat(manifold_dimension(G), σ), BG)
    cov = get_covariance_at(noise, x, BM)
    computed = ManifoldNormal.get_lie_covariance_at(noise, x, BG)
    @test isapprox(ManifoldNormal.get_lie_covariance_at(noise, x), computed)
    expected = PDMats.ScalMat(3, σ)
    @test isapprox(computed, expected)
    @test_throws ErrorException ManifoldNormal.get_lie_covariance_at(noise, [0,0,0])
    @test_throws ErrorException ManifoldNormal.get_lie_covariance_at(noise, [1, 0, 0], DefaultOrthonormalBasis())
    @test cov == PDMats.ScalMat(2, σ)

    nn = NoActionNoise(A)
    Z = ManifoldNormal.get_lie_covariance_at(nn, x, BG)
    lie_cov = ManifoldNormal.get_lie_covariance_at(noise, x, BG)
    new_lie_cov = lie_cov + Z
    @test isapprox(lie_cov, new_lie_cov)
end

"""
    check_adjoint_noise_involution(noise, χ)

When applied twice, `adjoint_noise` returns the same noise.
This is checked by computing the covariance at the point `χ`.
"""
check_adjoint_noise_involution(noise, χ) = begin
    n_ = adjoint_noise(noise)
    n__ = adjoint_noise(n_)
    computed = ManifoldNormal.get_lie_covariance_at(n__, χ)
    expected = ManifoldNormal.get_lie_covariance_at(noise, χ)
    return isapprox(computed, expected)
end

@testset for G in [
    SpecialOrthogonal(3)
]
    A = GroupOperationAction(G, (LeftAction(), LeftSide()))
    d = manifold_dimension(G)
    M = randn(rng, (d, d))
    Σ = PDMats.AbstractPDMat(M * M')
    @testset for noise in
        [
            ActionNoise(A, Σ),
            NoActionNoise(A),
        ]
        n_ = adjoint_noise(noise)
        n__ = adjoint_noise(n_)
        χ1 = rand(rng, G)
        χ2 = rand(rng, G)
        # @test isapprox(n__.covariance(χ1), n__.covariance(χ2))
        @test isapprox(ManifoldNormal.get_lie_covariance_at(n__, χ1), ManifoldNormal.get_lie_covariance_at(n__, χ2))
        @test check_adjoint_noise_involution(n_, χ1)
    end
end
