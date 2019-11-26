using .LibLightGBM: BoosterHandle, LGBM_BoosterFree

include("dataset.jl")

mutable struct Booster
    handle::BoosterHandle
    train_set::Union{Dataset, Nothing}
    test_set::Union{Dataset, Nothing}

    function Booster(handle::BoosterHandle, train_set :: Union{Nothing, Dataset}, test_set :: Union{Nothing, Dataset})
        bst = new(handle, train_set, test_set)
        finalizer(Booster_finalizer,bst)
        return bst
    end

    function Booster_finalizer(bst::Booster)
        if bst.handle != C_NULL
            LGBM_BoosterFree(bst.handle)
            bst.handle = C_NULL
        end
    end
end

function Booster()
    return Booster(C_NULL, nothing, nothing)
end

function Booster(handle::BoosterHandle)
    return Booster(handle, nothing, nothing)
end

mutable struct LGBM{T}
    booster::Booster
    model::Vector{String}
    application::String

    num_iterations::Int
    learning_rate::Float64
    num_leaves::Int
    max_depth::Int
    tree_learner::String
    num_threads::Int
    histogram_pool_size::Float64

    min_data_in_leaf::Int
    min_sum_hessian_in_leaf::Float64
    lambda_l1::Float64
    lambda_l2::Float64
    min_gain_to_split::Float64
    feature_fraction::Float64
    feature_fraction_seed::Int
    bagging_fraction::Float64
    bagging_freq::Int
    bagging_seed::Int
    early_stopping_round::Int

    max_bin::Int
    data_random_seed::Int
    init_score::String
    is_sparse::Bool
    save_binary::Bool
    categorical_feature::Vector{Int}

    sigmoid::Float64
    is_unbalance::Bool

    metric::Vector{String}
    metric_freq::Int
    is_training_metric::Bool
    ndcg_at::Vector{Int}

    num_machines::Int
    local_listen_port::Int
    time_out::Int
    machine_list_file::String

    num_class::Int
    device_type::String
end

"""
    LGBM(; [num_iterations = 10,
                      learning_rate = .1,
                      num_leaves = 127,
                      max_depth = -1,
                      tree_learner = \"serial\",
                      num_threads = Sys.CPU_THREADS,
                      histogram_pool_size = -1.,
                      min_data_in_leaf = 100,
                      min_sum_hessian_in_leaf = 10.,
                      lambda_l1 = 0.,
                      lambda_l2 = 0.,
                      min_gain_to_split = 0.,
                      feature_fraction = 1.,
                      feature_fraction_seed = 2,
                      bagging_fraction = 1.,
                      bagging_freq = 0,
                      bagging_seed = 3,
                      early_stopping_round = 0,
                      max_bin = 255,
                      data_random_seed = 1,
                      init_score = \"\",
                      is_sparse = true,
                      save_binary = false,
                      categorical_feature = Int[],
                      sigmoid = 1.,
                      is_unbalance = false,
                      metric = [\"l2\"],
                      metric_freq = 1,
                      is_training_metric = false,
                      ndcg_at = Int[],
                      num_machines = 1,
                      local_listen_port = 12400,
                      time_out = 120,
                      machine_list_file = \"\",
                     device_type=\"cpu\"])

Return a LGBMRegression estimator.
"""

@enum BoosterType Binary Regression Multiclass

function LGBM(T :: BoosterType ; num_iterations = 10,
                        learning_rate = .1,
                        num_leaves = 127,
                        max_depth = -1,
                        tree_learner = "serial",
                        num_threads = Sys.CPU_THREADS,
                        histogram_pool_size = -1.,
                        min_data_in_leaf = 100,
                        min_sum_hessian_in_leaf = 10.,
                        lambda_l1 = 0.,
                        lambda_l2 = 0.,
                        min_gain_to_split = 0.,
                        feature_fraction = 1.,
                        feature_fraction_seed = 2,
                        bagging_fraction = 1.,
                        bagging_freq = 0,
                        bagging_seed = 3,
                        early_stopping_round = 0,
                        max_bin = 255,
                        data_random_seed = 1,
                        init_score = "",
                        is_sparse = true,
                        save_binary = false,
                        categorical_feature = Int[],
                        sigmoid = 1.,
                        is_unbalance = false,
                        metric = ["l2"],
                        metric_freq = 1,
                        is_training_metric = false,
                        ndcg_at = Int[],
                        num_machines = 1,
                        local_listen_port = 12400,
                        time_out = 120,
                        machine_list_file = "",
                        device_type="cpu") 

    @assert(in(tree_learner, ("serial", "feature", "data")),
            "Unknown tree_learner, got $tree_learner")
    foreach(m -> @assert(in(m, ("l1", "l2", "ndcg", "auc", "binary_logloss", "binary_error",
                                "multi_logloss", "multi_error")),
                         "Unknown metric, got $m"), metric)

    return LGBM{T}(Booster() , String[], "regression", num_iterations, learning_rate, num_leaves,
                          max_depth, tree_learner, num_threads, histogram_pool_size,
                          min_data_in_leaf, min_sum_hessian_in_leaf, lambda_l1, lambda_l2,
                          min_gain_to_split, feature_fraction, feature_fraction_seed,
                          bagging_fraction, bagging_freq, bagging_seed, early_stopping_round,
                          max_bin, data_random_seed, init_score,
                          is_sparse, save_binary, categorical_feature,
                          sigmoid, is_unbalance, metric, metric_freq,
                          is_training_metric, ndcg_at, num_machines, local_listen_port, time_out,
                          machine_list_file,1,device_type)
end

LGBMRegression(kwargs...) = LGBM(Regression, kwargs...)
LGBMBinary(;kwargs...) = LGBM(Binary, kwargs...)
LGBMMulticlass(;kwargs...) = LGBM(Multiclass, kwargs...)

const BOOSTERPARAMS = [:application, :learning_rate, :num_leaves, :max_depth, :tree_learner,
                       :num_threads, :histogram_pool_size, :min_data_in_leaf,
                       :min_sum_hessian_in_leaf, :lambda_l1, :lambda_l2, :min_gain_to_split,
                       :feature_fraction, :feature_fraction_seed, :bagging_fraction,
                       :bagging_freq, :bagging_seed, :early_stopping_round, :sigmoid,
                       :is_unbalance, :metric, :is_training_metric, :ndcg_at, :num_machines,
                       :local_listen_port, :time_out, :machine_list_file, :num_class,:device]


# TODO add verboisty flag verboisty=1  #or some int
# TODO add epochs and early_stopping_round
function train!(model :: LGBM , dataset :: Dataset) where T <: BoosterType
    if model.booster.handle == C_NULL
        create_booster!(model, dataset)
    end

    success = Ref{Cint}()
    # Doesn't seem to be working

    LibLightGBM.LGBM_BoosterUpdateOneIter(model.booster.handle, success)
    success[] 
end


function create_booster!(model::LGBM, dataset::Dataset)
    booster_handle = Ref{BoosterHandle}()
    parameters = stringifyparams(model, BOOSTERPARAMS)
    LibLightGBM.LGBM_BoosterCreate(
        dataset.handle,
        parameters,
        booster_handle
    ) |> maybe_error
    model.booster = Booster(booster_handle[])
    model.booster.train_set = dataset
    model
end

# TODO add create from file
function create_booster!(filename::String)
    #LGBM_BoosterCreateFromModelfile
end

function get_current_iter(model::LGBM)
    if model.booster.handle == C_NULL
        0
    else
        n = Ref{Cint}()
        LibLightGBM.LGBM_BoosterGetCurrentIteration(model.booster.handle, n) |> maybe_error
        n[]
    end
end

"Get number of evaluation datasets "
function get_eval_counts(model::LGBM)
    if model.booster.handle == C_NULL
        0
    else
        n = Ref{Cint}()
        LibLightGBM.LGBM_BoosterGetEvalCounts(model.booster.handle, n) |> maybe_error
        n[]
    end
end

function get_eval_names(model::LGBM)

    # Doesn't this need to be here?
    if model.booster.handle == C_NULL
        throw(BoosterNotTrained("Need to train model first"))
    end

    n_metrics = LiteGBM.get_eval_counts(model)
    out_len = Ref{Cint}()
    out_strs = [ " " ^ 256 for _ in 1:n_metrics]
    LibLightGBM.LGBM_BoosterGetEvalNames(model.booster.handle, out_len, out_strs) |> maybe_error
    [string(split(s, "\0")[1]) for s in out_strs ]
end

"Returns the evaluation metrics as a dict"
function get_eval(model :: LGBM, dataset_type = :train)

    if model.booster.handle == C_NULL
        throw(BoosterNotTrained("Need to train model first"))
    end

    data_idx = if dataset_type == :train 
        isnothing(model.booster.train_set) && error("Model has no training set")
        0
    elseif dataset_type == :valid
        isnothing(model.booster.test_set) && error("Model has no test set")
        1
    else
        error("Invalid dataset type use either :train of :valid")
    end

    metrics = get_eval_names(model)
    n = length(metrics)
    out_results = Array{Float64}(undef, n)
    out_len = Ref{Cint}()
    LibLightGBM.LGBM_BoosterGetEval(model.booster.handle, data_idx, out_results ,out_results)
    Dict(zip(metrics, out_results))
end

"""
Saves the model to the file

* start_iter - start index of the iteration to be saved
* num_iter - the number of iterations to save

"""
function save(model :: LGBM, filename :: AbstractString, start_iter = 0, num_iters = 0)
    if model.booster.handle == C_NULL
        throw(BoosterNotTrained("Need to train model first"))
    end

    n_iters = get_current_iter(model)

    if start_iter + num_iters > n_iters
        error("Number of iterations is to many total is $n_iters. Tried to save $start_iter to $(start_iter + num_iters)")
    end

    if ispath(filename) 
        error("$filename already exists")
    end

    result = LibLightGBM.LGBM_BoosterSaveModel(model.booster.handle, start_iter, num_iters, filename)

    if result == 0
        return 
    else
        error("Unknown error saving model - $(LibLightGBM.LGBM_GetLastError() |> unsafe_string)")
    end

end
   
function load(filename :: AbstractString)
    out_num_iter = Ref{Cint}()
    booster = Booster()
    handle = Ref{BoosterHandle}()

    if !isfile(filename)
        error("$filename doens't exists")
    end

    result = LibLightGBM.LGBM_BoosterCreateFromModelfile(filename, out_num_iter, handle)
    
    if result == 0
        booster.handle = handle[]
        num_class = Ref{Cint}()
        LibLightGBM.LGBM_BoosterGetNumClasses(booster.handle,  num_class)

        # TODO need to load other relevant parameters!
        # and support other objectives e.g classification 
        parameters = load_parameters(filename)

        deleteat!

        model = LGBM(Regression; parameters...)
        model.booster = booster
        model.num_class = num_class[]

        return booster
    else
        error("Error loading model")
    end

end

function load_parameters(filename :: AbstractString)

    s = read(filename, String)
    d = Dict{Symbol, Any}()
    fields = Dict(zip(fieldnames(LGBM), fieldtypes(LGBM)))

    # TODO handle parameters other than Float or Int
    for m in eachmatch(r"\[([\w_)]+): (.+)\]", s)
        key = Symbol(m[1])
        if haskey(fields,key) && fields[key] <: Number
            val = parse(Float64,m[2])
            d[key] = convert(fields[key], val)
        end
    end

    delete!(d, :num_class)

    NamedTuple{Tuple(keys(d))}(values(d))
end

function parsenum(s)
    try 
        return parse(Int,s)
    catch
        return parse(Float64,s)
    end
end

struct BoosterNotTrained <: Exception
    s :: String
end
