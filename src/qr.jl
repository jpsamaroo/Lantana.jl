export qr!

struct QRAlgorithm{F<:Function} <: LinearAlgebraAlgorithm
    func::F
    oop::Bool
    A_type::DataType
    accum_type::DataType
    accuracy::Float64
end

const QR_ALGORITHMS = Vector{QRAlgorithm}()

function register_qr_algorithm!(func::Function;
                                      oop::Bool=false,
                                      A_type::DataType,
                                      accum_type::DataType,
                                      accuracy::Float64)
    push!(QR_ALGORITHMS, QRAlgorithm(func, oop, A_type, accum_type, accuracy))
end

struct QRCall <: LinearAlgebraCall
    A_type::DataType
    accum_type::DataType
    accuracy::Float64
    algorithm::Union{QRAlgorithm, Nothing}
end

function select_algorithm(call::QRCall)
    for alg in QR_ALGORITHMS
        if alg.A_type == call.A_type &&
            alg.accum_type == call.accum_type &&
            alg.accuracy >= call.accuracy
            return alg
        end
    end

    # FIXME: Call a fallback algorithm
    error("No algorithm found for call $call")
end

function qr!(A;
             accum_type::DataType=eltype(A),
             accuracy::Float64=0.0,
             algorithm::Union{QRAlgorithm, Nothing}=nothing)
    call = QRCall(
        typeof(A),
        accum_type,
        accuracy,
        algorithm
    )
    alg = select_algorithm(call)
    return alg.func(A)
end