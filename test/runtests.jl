using Lantana
using Test

@testset "Registered Algorithms" begin
    @test length(Lantana.MATRIX_MULTIPLY_ALGORITHMS) > 0
    @test length(Lantana.CHOLESKY_ALGORITHMS) > 0
    @test length(Lantana.LU_ALGORITHMS) > 0
    @test length(Lantana.QR_ALGORITHMS) > 0
    @test length(Lantana.TRSM_ALGORITHMS) > 0
end
