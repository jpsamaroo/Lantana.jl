module CUDAExt

import Lantana
import CUDA

function __init__()
    # Matrix multiply
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f!(C, A, B, alpha, beta) = CUDA.CUBLAS.gemm!(
            'N',
            'N',
            alpha,
            A,
            B,
            beta,
            C,
        )
        register_matrix_multiply_algorithm!(f!;
                                            C_type=CuMatrix{T},
                                            A_type=CuMatrix{T},
                                            B_type=CuMatrix{T},
                                            accum_type=T,
                                            accuracy=0.0)
    end

    # Cholesky
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f!(A) = CUDA.CUSOLVER.potrf!('U', A)
        register_cholesky_algorithm!(f!;
                                     A_type=CuMatrix{T},
                                     accum_type=T,
                                     accuracy=0.0)
    end

    # LU
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f!(A) = CUDA.CUSOLVER.getrf!(A)
        register_lu_algorithm!(f!;
                               A_type=CuMatrix{T},
                               accum_type=T,
                               accuracy=0.0)
    end

    # QR
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f!(A) = CUDA.CUSOLVER.geqrf!(A)
        register_qr_algorithm!(f!;
                               A_type=CuMatrix{T},
                               accum_type=T,
                               accuracy=0.0)
    end

    # TRSM
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f!(A, B) = CUDA.CUSOLVER.trsm!('L', 'U', 'N', 'N', 1.0, A, B)
        register_trsm_algorithm!(f!;
                                 A_type=CuMatrix{T},
                                 B_type=CuMatrix{T},
                                 accum_type=T,
                                 accuracy=0.0)
    end
end

end # module CUDAExt