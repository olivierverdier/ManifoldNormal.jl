using Manifolds

import Random
using Distributions
import PDMats

import LinearAlgebra

rng = Random.default_rng()

@testset "Rotation action" begin
    dim = 3
    G = SpecialOrthogonal(dim)
    A = RotationAction(Euclidean(dim), G)
    M = group_manifold(A)
    μ = zeros(manifold_dimension(M))
    # μ = rand(rng, M)
    dG = manifold_dimension(G)
    mat = randn(rng, dG, dG)
    Σ = PDMats.PDMat(mat*mat')
    D = ActionDistribution(A, μ, Σ)
    # possible to initialise with a matrix
    D_ = ActionDistribution(A, μ, mat*mat')
    @test_throws LinearAlgebra.PosDefException ActionDistribution(A, μ, zeros(dG,dG))
    apply!(get_action(D), similar(D.μ), identity_element(G), D.μ)
    rand(D)
    @test_throws TypeError ActionDistribution(switch_direction(A), μ, Σ)
end

@testset "ActionDistribution" begin
    G = SpecialOrthogonal(3)
    M = Sphere(2)
    A = Manifolds.ColumnwiseMultiplicationAction(M, G)
    μ = [1., 0, 0]
    Σ = PDMats.ScalMat(3, 1.)
    @test isapprox(Distributions.cov(ActionDistribution(A, μ, 1.)), Σ)
    B = DefaultOrthonormalBasis()
    dist = ActionDistribution(A, μ, Σ, B)
    x = rand(rng, dist)
    rand(dist)
    @test scaled_distance(dist, [0., 1, 0]) == scaled_distance(dist, [0, 0, 1.])
    @test update_mean(dist, mean(dist)) == dist
    @test update_mean_cov(dist, mean(dist), cov(dist)) == dist
end

@testset "action_noise" begin
    G = Orthogonal(3)
    M = Sphere(2)
    A = Manifolds.ColumnwiseMultiplicationAction(M, G)
    μ = [0, 0, 1.0]
    Σ(p) = PDMats.PDiagMat(abs.(p))
    B = DefaultOrthonormalBasis()
    noise = ActionNoise(A, Σ, B)
    @test noise(rng, μ) ≈ μ

    @testset "Constant Noise" begin
        cnoise = ManifoldNormal.constant_noise_at(noise, μ)
        x = [1,0,0]
        not_expected = ManifoldNormal.get_lie_covariance_at(noise, x)
        expected = ManifoldNormal.get_lie_covariance_at(noise, μ)
        computed = ManifoldNormal.get_lie_covariance_at(cnoise, x)
        @test computed ≠ not_expected
        @test computed ≈ expected
    end

    dist = ActionDistribution(μ, noise)
    @test !(noise isa ActionNoise{<:Any, <:Returns})
    @test action_noise(dist) isa ActionNoise{<:Any, <:Returns}
    @test length(dist) == 2
    @test rand(rng, dist) ≈ μ
    @test startswith(sprint(show, dist), "ActionDistribution")

    @testset "random_vector_lie" begin
        rnd_lie = ManifoldNormal.random_vector_lie(rng, dist)
        coords = get_coordinates(G, identity_element(G), rnd_lie, ManifoldNormal.get_lie_basis(dist))
        @test all(first(coords, 2) .≈ 0)
    end
end

