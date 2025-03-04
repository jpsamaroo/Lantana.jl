export cholesky!

struct CholeskyAlgorithm{F<:Function} <: LinearAlgebraAlgorithm
    func::F
    oop::Bool
    A_type::DataType
    accum_type::DataType
    accuracy::Float64
end

const CHOLESKY_ALGORITHMS = Vector{CholeskyAlgorithm}()

function register_cholesky_algorithm!(func::Function;
                                      oop::Bool=false,
                                      A_type::DataType,
                                      accum_type::DataType,
                                      accuracy::Float64)
    push!(CHOLESKY_ALGORITHMS, CholeskyAlgorithm(func, oop, A_type, accum_type, accuracy))
end

struct CholeskyCall <: LinearAlgebraCall
    A_type::DataType
    accum_type::DataType
    accuracy::Float64
    algorithm::Union{CholeskyAlgorithm, Nothing}
end

function select_algorithm(call::CholeskyCall)
    for alg in CHOLESKY_ALGORITHMS
        if alg.A_type == call.A_type &&
            alg.accum_type == call.accum_type &&
            alg.accuracy >= call.accuracy
            return alg
        end
    end

    # FIXME: Call a fallback algorithm
    error("No algorithm found for call $call")
end

function cholesky!(A;
                   accum_type::DataType=eltype(A),
                   accuracy::Float64=0.0,
                   algorithm::Union{CholeskyAlgorithm, Nothing}=nothing)
    call = CholeskyCall(
        typeof(A),
        accum_type,
        accuracy,
        algorithm
    )
    alg = select_algorithm(call)
    return alg.func(A)
end