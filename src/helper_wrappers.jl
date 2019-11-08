import .LibLightGBM
import .LibLightGBM: C_API_DTYPE_FLOAT32, C_API_DTYPE_FLOAT64, C_API_DTYPE_INT32, C_API_DTYPE_INT64


function jltype_to_lgbmid(datatype::Type)
    if datatype == Float32
        return C_API_DTYPE_FLOAT32
    elseif datatype == Float64
        return C_API_DTYPE_FLOAT64
    elseif datatype == Int32
        return C_API_DTYPE_INT32
    elseif datatype == Int64
        return C_API_DTYPE_INT64
    else
        error("unsupported datatype, got ", datatype)
    end
end

function lgbmid_to_jltype(id::Integer)
    if id == C_API_DTYPE_FLOAT32
        return Float32
    elseif id == C_API_DTYPE_FLOAT64
        return Float64
    elseif id == C_API_DTYPE_INT32
        return Int32
    elseif id == C_API_DTYPE_INT64
        return Int64
    else
        error("unknown LightGBM return type id, got ", id)
    end
end

function LGBM_DatasetCreateFromMat(data::Matrix{T}, parameters::String, is_row_major::Bool = false) where T<:Union{Float32,Float64}
    lgbm_data_type = jltype_to_lgbmid(T)
    nrow, ncol = ifelse(is_row_major, reverse(size(data)), size(data))
    out = Ref{DatasetHandle}()
    LibLightGBM.LGBM_DatasetCreateFromMat(data, lgbm_data_type, nrow, ncol, is_row_major, parameters, C_NULL, out)
    return Dataset(out[])
end

function _LGBM_DatasetSetField(ds::Dataset, field_name::String,
                                                        field_data::Vector{T}) where T <:Union{Float32,Float64,Int32}
    data_type = jltype_to_lgbmid(T)
    num_element = length(field_data)
    LibLightGBM.LGBM_DatasetSetField(
        ds.handle,
        field_name,
        field_data,
        num_element,
        data_type
    )
    return nothing
end

function LGBM_DatasetSetField(ds::Dataset, field_name::String, field_data::Vector{Float32})
    if field_name == "label" || field_name == "weight"
        _LGBM_DatasetSetField(ds, field_name, field_data)
    else
        _LGBM_DatasetSetField(ds, field_name, convert(Vector{Int32}, field_data))
    end
    return nothing
end

function LGBM_DatasetSetField(ds::Dataset, field_name::String, field_data::Vector{Int32})
    if field_name == "group" || field_name == "group_id"
        _LGBM_DatasetSetField(ds, field_name, field_data)
    else
        _LGBM_DatasetSetField(ds, field_name, convert(Vector{Float32}, field_data))
    end
    return nothing
end

function LGBM_DatasetSetField(ds::Dataset, field_name::String, field_data::Vector{T}) where T<:Real
    if field_name == "label" || field_name == "weight"
        _LGBM_DatasetSetField(ds, field_name, convert(Vector{Float32}, field_data))
    elseif field_name == "init_score"
        _LGBM_DatasetSetField(ds, field_name, convert(Vector{Float64}, field_data))
    elseif field_name == "group" || field_name == "group_id"
        _LGBM_DatasetSetField(ds, field_name, convert(Vector{Int32}, field_data))
    end
    return nothing
end

function LGBM_BoosterCreate(train_data::Dataset, parameters::String)
    out = Ref{BoosterHandle}()
    LibLightGBM.LGBM_BoosterCreate(
              train_data.handle,
              parameters,
              out, )
    return Booster(out[], [train_data])
end

function LGBM_BoosterAddValidData(bst::Booster, valid_data::Dataset)
    LibLightGBM.LGBM_BoosterAddValidData(
              bst.handle ,
              valid_data.handle 
    )
    push!(bst.datasets, valid_data)
    return nothing
end

function LGBM_BoosterGetEvalNames(bst::Booster)
    out_len = Ref{Cint}()
    n_metrics = LGBM_BoosterGetEvalCounts(bst)
    out_strs = [Vector{UInt8}(undef, 256) for i in 1:n_metrics]
    LibLightGBM.LGBM_BoosterGetEvalNames(
        bst.handle,
        out_len ,
        out_strs 
    )
    jl_out_strs = [unsafe_string(pointer(out_str)) for out_str in out_strs[1:out_len[]]]
    return jl_out_strs
end

function LGBM_BoosterGetEvalCounts(bst::Booster)
    out_len = Ref{Cint}()
    LibLightGBM.LGBM_BoosterGetEvalCounts(
        bst.handle,
        out_len,
    )
    return out_len[]
end

function LGBM_BoosterGetEvalNames(bst::Booster)
    out_len = Ref{Cint}()
    n_metrics = LGBM_BoosterGetEvalCounts(bst)
    out_strs = [Vector{Cchar}(undef, 256) for i in 1:n_metrics]
    LibLightGBM.LGBM_BoosterGetEvalNames(
        bst.handle ,
        out_len,
        out_strs,
    )
    jl_out_strs = [unsafe_string(pointer(out_str)) for out_str in out_strs[1:out_len[]]]
    return jl_out_strs
end

function LGBM_BoosterUpdateOneIter(bst::Booster)
    is_finished = Ref{Cint}()
    LibLightGBM.LGBM_BoosterUpdateOneIter(
        bst.handle ,
        is_finished 
    )
    return is_finished[]
end

# function LGBM_BoosterGetEval(bst::Booster, data::Integer)
#     n_metrics = LGBM_BoosterGetEvalCounts(bst)
#     out_results = Array{Cdouble}(undef, n_metrics)
#     out_len = Ref{Cint}()
#     LibLightGBM.LGBM_BoosterGetEval(
#         bst.handle, 
#         data ,
#         out_len ,
#         out_results 
#     )
#     return out_results[1:out_len[]]
# end

macro lightgbm(f, params...)
    return quote
        err = ccall(($f, LibLightGBM.liblightgbm), Cint,
                    ($((esc(i.args[end]) for i in params)...),),
                    $((esc(i.args[end - 1]) for i in params)...))
        if err != 0
            msg = unsafe_string(LibLightGBM.LGBM_GetLastError)
            error("call to LightGBM's ", string($(esc(f))), " failed: ", msg)
        end
    end
end
function LGBM_BoosterGetEvalNames(bst::Booster)
    out_len = Ref{Cint}()
    n_metrics = LGBM_BoosterGetEvalCounts(bst)
    out_strs = [Vector{UInt8}(undef, 256) for i in 1:n_metrics]
    @lightgbm(:LGBM_BoosterGetEvalNames,
              bst.handle => BoosterHandle,
              out_len => Ref{Cint},
              out_strs => Ref{Ptr{UInt8}})
    jl_out_strs = [unsafe_string(pointer(out_str)) for out_str in out_strs[1:out_len[]]]
    return jl_out_strs
end