export mul!

struct MatrixMultiplyAlgorithm{F<:Function} <: LinearAlgebraAlgorithm
    func::F
    oop::Bool
    C_type::DataType
    A_type::DataType
    B_type::DataType
    accum_type::DataType
    accuracy::Float64
end

const MATRIX_MULTIPLY_ALGORITHMS = Vector{MatrixMultiplyAlgorithm}()

function register_matrix_multiply_algorithm!(func::Function;
                                             oop::Bool=false,
                                             C_type::DataType,
                                             A_type::DataType,
                                             B_type::DataType,
                                             accum_type::DataType,
                                             accuracy::Float64)
    push!(MATRIX_MULTIPLY_ALGORITHMS, MatrixMultiplyAlgorithm(func, oop, C_type, A_type, B_type, accum_type, accuracy))
end

struct MatrixMultiplyCall <: LinearAlgebraCall
    C_type::DataType
    A_type::DataType
    B_type::DataType
    accum_type::DataType
    accuracy::Float64
    algorithm::Union{MatrixMultiplyAlgorithm, Nothing}
end

function select_algorithm(call::MatrixMultiplyCall)
    for alg in MATRIX_MULTIPLY_ALGORITHMS
        if alg.C_type == call.C_type &&
            alg.A_type == call.A_type &&
            alg.B_type == call.B_type &&
            alg.accum_type == call.accum_type &&
            alg.accuracy >= call.accuracy
            return alg
        end
    end

    # FIXME: Call a fallback algorithm
    error("No algorithm found for call $call")
end

function mul!(C, A, B, alpha, beta;
              accum_type::DataType=eltype(C),
              accuracy::Float64=0.0,
              algorithm::Union{MatrixMultiplyAlgorithm, Nothing}=nothing)
    call = MatrixMultiplyCall(
        typeof(C),
        typeof(A),
        typeof(B),
        accum_type,
        accuracy,
        algorithm
    )
    alg = select_algorithm(call)
    return alg.func(C, A, B, alpha, beta)
end
mul!(C, A, B; alpha=1.0, beta=0.0) = mul!(C, A, B, alpha, beta)