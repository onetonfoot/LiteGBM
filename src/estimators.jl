using .LibLightGBM: BoosterHandle, LGBM_BoosterFree

include("dataset.jl")

mutable struct Booster
    handle::BoosterHandle
    datasets::Vector{Dataset}

    function Booster(handle::BoosterHandle, datasets::Vector{Dataset})
        bst = new(handle, datasets)
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
    return Booster(C_NULL, Dataset[])
end

function Booster(handle::BoosterHandle)
    return Booster(handle, Dataset[])
end

# TODO shouldn't the dataset parameters be detached from the booster ?!
# Creating the struct should also create the handle to the booster model
# Not during the fit method

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

@enum BoosterType binary regression multiclass

function LGBM(t::BoosterType; num_iterations = 10,
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

    return LGBM{t}(Booster(), String[], "regression", num_iterations, learning_rate, num_leaves,
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

LGBMRegression(kwargs...) = LGBM(regression, kwargs...)
LGBMBinary(;kwargs...) = LGBM(binary, kwargs...)
LGBMMulticlass(;kwargs...) = LGBM(multiclass, kwargs...)

const BOOSTERPARAMS = [:application, :learning_rate, :num_leaves, :max_depth, :tree_learner,
                       :num_threads, :histogram_pool_size, :min_data_in_leaf,
                       :min_sum_hessian_in_leaf, :lambda_l1, :lambda_l2, :min_gain_to_split,
                       :feature_fraction, :feature_fraction_seed, :bagging_fraction,
                       :bagging_freq, :bagging_seed, :early_stopping_round, :sigmoid,
                       :is_unbalance, :metric, :is_training_metric, :ndcg_at, :num_machines,
                       :local_listen_port, :time_out, :machine_list_file, :num_class,:device]

function train!(model, dataset)
    if model.booster.handle == C_NULL
        # TODO add verboisty flag verboisty=1  #or some int
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
    n_metrics = LiteGBM.get_eval_counts(model)
    out_len = Ref{Cint}()
    out_strs = [ " " ^ 256 for _ in 1:n_metrics]
    LibLightGBM.LGBM_BoosterGetEvalNames(model.booster.handle, out_len, out_strs) |> maybe_error
    [string(split(s, "\0")[1]) for s in out_strs ]
end

"Returns the evaluation metrics as a dict"
function get_eval(model, dataset_type = :train)
    if model.booster.handle == C_NULL
        error("No booster avaliable please train model first")
    end

    data_idx = if dataset_type == :valid
        if get_eval_counts(model) >= 1
            1
        else
            @warn "No :valid dataset using :train"
            0
        end
    elseif dataset_type == :train
        0
    else
        @warn "Unknown evaluation type $dataset_type using :train"
        0
    end

    metrics = get_eval_names(model)
    n = length(metrics)
    out_results = Array{Float64}(undef, n)
    out_len = Ref{Cint}()
    LibLightGBM.LGBM_BoosterGetEval(model.booster.handle, data_idx, out_results ,out_results)
    Dict(zip(metrics, out_results))
end
   


