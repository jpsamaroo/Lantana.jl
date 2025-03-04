export trsm!

struct TRSMAlgorithm{F<:Function} <: LinearAlgebraAlgorithm
    func::F
    oop::Bool
    A_type::DataType
    B_type::DataType
    accum_type::DataType
    accuracy::Float64
end

const TRSM_ALGORITHMS = Vector{TRSMAlgorithm}()

function register_trsm_algorithm!(func::Function;
                                  oop::Bool=false,
                                  A_type::DataType,
                                  B_type::DataType,
                                  accum_type::DataType,
                                  accuracy::Float64)
    push!(TRSM_ALGORITHMS, TRSMAlgorithm(func, oop, A_type, B_type, accum_type, accuracy))
end

struct TRSMCall <: LinearAlgebraCall
    A_type::DataType
    B_type::DataType
    accum_type::DataType
    accuracy::Float64
    algorithm::Union{TRSMAlgorithm, Nothing}
end

function select_algorithm(call::TRSMCall)
    for alg in TRSM_ALGORITHMS
        if alg.A_type == call.A_type &&
            alg.B_type == call.B_type &&
            alg.accum_type == call.accum_type &&
            alg.accuracy >= call.accuracy
            return alg
        end
    end

    # FIXME: Call a fallback algorithm
    error("No algorithm found for call $call")
end

function trsm!(A, B;
               accum_type::DataType=eltype(A),
               accuracy::Float64=0.0,
               algorithm::Union{TRSMAlgorithm, Nothing}=nothing)
    call = TRSMCall(
        typeof(A),
        typeof(B),
        accum_type,
        accuracy,
        algorithm
    )
    alg = select_algorithm(call)
    return alg.func(A, B)
end