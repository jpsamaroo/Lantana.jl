export lu!

struct LUAlgorithm{F<:Function} <: LinearAlgebraAlgorithm
    func::F
    oop::Bool
    A_type::DataType
    accum_type::DataType
    accuracy::Float64
end

const LU_ALGORITHMS = Vector{LUAlgorithm}()

function register_lu_algorithm!(func::Function;
                                      oop::Bool=false,
                                      A_type::DataType,
                                      accum_type::DataType,
                                      accuracy::Float64)
    push!(LU_ALGORITHMS, LUAlgorithm(func, oop, A_type, accum_type, accuracy))
end

struct LUCall <: LinearAlgebraCall
    A_type::DataType
    accum_type::DataType
    accuracy::Float64
    algorithm::Union{LUAlgorithm, Nothing}
end

function select_algorithm(call::LUCall)
    for alg in LU_ALGORITHMS
        if alg.A_type == call.A_type &&
            alg.accum_type == call.accum_type &&
            alg.accuracy >= call.accuracy
            return alg
        end
    end

    # FIXME: Call a fallback algorithm
    error("No algorithm found for call $call")
end

function lu!(A;
             accum_type::DataType=eltype(A),
             accuracy::Float64=0.0,
             algorithm::Union{LUAlgorithm, Nothing}=nothing)
    call = LUCall(
        typeof(A),
        accum_type,
        accuracy,
        algorithm
    )
    alg = select_algorithm(call)
    return alg.func(A)
end