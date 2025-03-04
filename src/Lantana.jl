module Lantana

import LinearAlgebra as LA

abstract type LinearAlgebraAlgorithm end

abstract type LinearAlgebraCall end

include("matrix_multiply.jl")
include("cholesky.jl")
include("lu.jl")
include("qr.jl")
include("triangular_solve.jl")

function __init__()
    # Matrix multiply
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f!(C, A, B, alpha, beta) = LA.BLAS.gemm!(
            'N',
            'N',
            alpha,
            A,
            B,
            beta,
            C,
        )
        register_matrix_multiply_algorithm!(f!;
                                            C_type=Matrix{T},
                                            A_type=Matrix{T},
                                            B_type=Matrix{T},
                                            accum_type=T,
                                            accuracy=0.0)
    end

    # Cholesky
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f!(A) = LA.LAPACK.potrf!('U', A)
        register_cholesky_algorithm!(f!;
                                     A_type=Matrix{T},
                                     accum_type=T,
                                     accuracy=0.0)
    end

    # LU
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f!(A) = LA.LAPACK.getrf!(A)
        register_lu_algorithm!(f!;
                               A_type=Matrix{T},
                               accum_type=T,
                               accuracy=0.0)
    end

    # QR
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f!(A) = LA.LAPACK.geqrf!(A)
        register_qr_algorithm!(f!;
                               A_type=Matrix{T},
                               accum_type=T,
                               accuracy=0.0)
    end

    # TRSM
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f!(A, B) = LA.BLAS.trsm!('L', 'U', 'N', 'N', 1.0, A, B)
        register_trsm_algorithm!(f!;
                                 A_type=Matrix{T},
                                 B_type=Matrix{T},
                                 accum_type=T,
                                 accuracy=0.0)
    end
end

end # module Lantana
