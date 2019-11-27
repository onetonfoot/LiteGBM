import .LibLightGBM
import .LibLightGBM: DatasetHandle, C_API_DTYPE_FLOAT32, C_API_DTYPE_FLOAT64, C_API_DTYPE_INT32, C_API_DTYPE_INT64

const DATASETPARAMS = [:is_sparse, :max_bin, :data_random_seed, :categorical_feature]
const INDEXPARAMS = [:categorical_feature]

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

mutable struct Dataset
    handle::DatasetHandle
    reference::DatasetHandle # If training should be C_NULL otherwise should reference training data (maybe use a parametric type instead)
    weight::Array{Float32}
    group::Array{Int32}
    init_score::Array{Float64}
    # silent
    feature_name::Array{Symbol}
    categorical_feature::Array{Int}
    # params 
    # free_raw_data
    # label 

    function Dataset(handle::DatasetHandle)
        ds = new(handle)
        finalizer(Dataset_finalizer,ds)
        return ds
    end

    function Dataset_finalizer(ds::Dataset)
        if ds.handle != C_NULL
            LibLightGBM.LGBM_DatasetFree(ds.handle) 
            ds.handle = C_NULL
        end
    end
end

function Dataset(X::Matrix{T}, y, train_dataset :: Union{Nothing, Dataset} = nothing) where T <: Union{Float32, Float64} 

    is_row_major = false
    nrows, ncols = size(X)
    data_type = jltype_to_lgbmid(T)
    ref = Ref{DatasetHandle}()

    train_ref = isnothing(train_dataset) ? C_NULL : train_dataset.handle

    # TODO add relevant dataset parameters 
    parameters = ""

    # TODO How to handle a matrix with mixed features? Strings Floats Ints 
    LibLightGBM.LGBM_DatasetCreateFromMat(
        X, data_type, nrows, ncols, is_row_major, parameters, train_ref, ref
    ) |> maybe_error

    dataset = Dataset(ref[])
    set_label(y, dataset.handle)
    return dataset
end

# TODO Should be able to pass a list of categorical_features
# rename add traning set
# Maybe rename to add parameters for something like that
function create_train_dataset(X::Matrix{T}, y; 
        weights=[],
        groups=[],
        init_score = [],
        ) where T <: Union{Float32, Float64}
    


    dataset = Dataset(X, y)

    for (data, fn) in zip([weights, groups, init_score], [set_weight, set_group, set_init_score])
        if !isempty(data)
            fn(data, dataset.handle)
        end
    end
    
    return dataset
end

function add_test_data!(model, test_set)
    if model.booster.handle == C_NULL
        error("Model has no associated booster yet. Please add a training dataset first!")
    end

    if isnothing(model.booster.train_set) 
        error("Need to add training before validation data first")
    end

    #Should assert that the columns match

    LibLightGBM.LGBM_BoosterAddValidData(model.booster.handle, test_set.handle) |> maybe_error
    model.booster.test_set = test_set
end


function set_group(groups, handle::DatasetHandle)
    groups = convert(Vector{Int32}, groups)
    LibLightGBM.LGBM_DatasetSetField(
        handle, "group", groups, length(groups), LibLightGBM.C_API_DTYPE_INT32
    ) |> maybe_error
end

function set_label(label, handle::DatasetHandle)
    label = convert(Vector{Float32}, label)
    LibLightGBM.LGBM_DatasetSetField(
        handle, "label", label, length(label), LibLightGBM.C_API_DTYPE_FLOAT32
    ) |> maybe_error
end

function set_weight(weight, handle::DatasetHandle)
    weight = convert(Vector{Float32}, weight)
    LibLightGBM.LGBM_DatasetSetField(
        handle, "weight", weight, length(weight), LibLightGBM.C_API_DTYPE_FLOAT32
    ) |> maybe_error
end

function set_init_score(init_score, handle::DatasetHandle)
    init_score = convert(Vector{Float64}, init_score)
    LibLightGBM.LGBM_DatasetSetField(
        handle, "init_score", init_score, length(init_score), LibLightGBM.C_API_DTYPE_FLOAT64
    ) |> maybe_error
end

function maybe_error(result)
    if result == 0
        nothing
    else
        error(LibLightGBM.LGBM_GetLastError() |> unsafe_string)
    end
end

function Base.size(dataset::Dataset)
    x, y = Ref{Cint}(), Ref{Cint}()
    LibLightGBM.LGBM_DatasetGetNumData(dataset.handle, x) |> maybe_error
    LibLightGBM.LGBM_DatasetGetNumFeature(dataset.handle, y) |> maybe_error
    (x[], y[])
end

# X::Union{String, Matrix, DataFrame, Vector{Matrix}} # For now just matrix
# y::Array{T}
# weights::Array{Float64} ?
# init_score ?
# feature_names::Union{Array{Int},Array{Symbol}} # defaults to auto
# categorical_features::Union{Array{Int},Array{Symbol}} # 

# * data (string, numpy array, pandas DataFrame, H2O DataTable's Frame, scipy.sparse or list of numpy arrays) – Data source of Dataset. If string, it represents the path to txt file.
# * label (list, numpy 1-D array, pandas Series / one-column DataFrame or None, optional (default=None)) – Label of the data.
# * reference (Dataset or None, optional (default=None)) – If this is Dataset for validation, training data should be used as reference.
# * weight (list, numpy 1-D array, pandas Series or None, optional (default=None)) – Weight for each instance.
# * group (list, numpy 1-D array, pandas Series or None, optional (default=None)) – Group/query size for Dataset.
# * init_score (list, numpy 1-D array, pandas Series or None, optional (default=None)) – Init score for Dataset.
# * silent (bool, optional (default=False)) – Whether to print messages during construction.
# * feature_name (list of strings or 'auto', optional (default="auto")) – Feature names. If ‘auto’ and data is pandas DataFrame, data columns names are used.
# * categorical_feature (list of strings or int, or 'auto', optional (default="auto")) – Categorical features. If list of int, interpreted as indices. If list of strings, interpreted as feature names (need to specify feature_name as well). If ‘auto’ and data is pandas DataFrame, pandas unordered categorical columns are used. All values in categorical features should be less than int32 max value (2147483647). Large values could be memory consuming. Consider using consecutive integers starting from zero. All negative values in categorical features will be treated as missing values. The output cannot be monotonically constrained with respect to a categorical feature.
# * params (dict or None, optional (default=None)) – Other parameters for Dataset.
# * free_raw_data (bool, optional (default=True)) – If True, raw data is freed after constructing inner Dataset.

function stringifyparams(estimator, params::Vector{Symbol})
    paramstring = ""
    n_params = length(params)
    valid_names = estimator |> typeof |> fieldnames |> collect

    # TODO deleted sigmoid parameter for non binary classifcation
    # idx = findfirst(==(:c), (:a,:b,:c))
    # deleteat!(valid_names, idx)

    for (param_idx, param_name) in enumerate(params)
        if in(param_name, valid_names)
            param_value = getfield(estimator, param_name)
            if !isempty(param_value)
                # Convert parameters that contain indices to C's zero-based indices.
                if in(param_name, INDEXPARAMS)
                    param_value -= 1
                end

                if typeof(param_value) <: Array
                    n_entries = length(param_value)
                    if n_entries >= 1
                        paramstring = string(paramstring, param_name, "=", param_value[1])
                        for entry_idx in 2:n_entries
                            paramstring = string(paramstring, ",", param_value[entry_idx])
                        end
                        paramstring = string(paramstring, " ")
                    end
                else
                    paramstring = string(paramstring, param_name, "=", param_value, " ")
                end
            end
        end
    end
    return paramstring[1:end - 1]
end