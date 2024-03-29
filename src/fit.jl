import Dates: DateTime, now
import Dates

include("helper_wrappers.jl")

const DATASETPARAMS = [:is_sparse, :max_bin, :data_random_seed, :categorical_feature]

const BOOSTERPARAMS = [:application, :learning_rate, :num_leaves, :max_depth, :tree_learner,
                       :num_threads, :histogram_pool_size, :min_data_in_leaf,
                       :min_sum_hessian_in_leaf, :lambda_l1, :lambda_l2, :min_gain_to_split,
                       :feature_fraction, :feature_fraction_seed, :bagging_fraction,
                       :bagging_freq, :bagging_seed, :early_stopping_round, :sigmoid,
                       :is_unbalance, :metric, :is_training_metric, :ndcg_at, :num_machines,
                       :local_listen_port, :time_out, :machine_list_file, :num_class,:device]

const INDEXPARAMS = [:categorical_feature]
const MAXIMIZE_METRICS = ["auc", "ndcg"]

function shrinkresults!(results, last_retained_iter::Integer)
    for test_key in keys(results)
        test = results[test_key]
        for metric_key in keys(test)
            test[metric_key] = test[metric_key][1:last_retained_iter]
        end
    end
    return nothing
end

"""
    fit(estimator, X, y[, test...]; [verbosity = 1, is_row_major = false])

Fit the `estimator` with features data `X` and label `y` using the X-y pairs in `test` as
validation sets.

Return a dictionary with an entry for each validation set. Each entry of the dictionary is another
dictionary with an entry for each validation metric in the `estimator`. Each of these entries is an
array that holds the validation metric's value at each iteration.

# Arguments
* `estimator::LGBMEstimator`: the estimator to be fit.
* `X::Matrix{TX<:Real}`: the features data.
* `y::Vector{Ty<:Real}`: the labels.
* `test::Tuple{Matrix{TX},Vector{Ty}}...`: optionally contains one or more tuples of X-y pairs of
    the same types as `X` and `y` that should be used as validation sets.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.
* `is_row_major::Bool`: keyword argument that indicates whether or not `X` is row-major. `true`
    indicates that it is row-major, `false` indicates that it is column-major (Julia's default).
* `weights::Vector{Tw<:Real}`: the training weights.
* `init_score::Vector{Ti<:Real}`: the init scores.
"""



function fit(estimator::LGBM, X::Matrix{TX},
    y::Vector{Ty}, test::Tuple{Matrix{TX},Vector{Ty}}...; verbosity::Integer = 1,
    is_row_major = false, weights::Vector{Tw} = Vector{Float32}(),
    init_score::Vector{Ti} = Vector{Float64}()) where {TX<:Real,Ty<:Real,Tw<:Real,Ti<:Real}

    start_time = now()

    @debug "Started creating LGBM training dataset\n"

    ds_parameters = stringifyparams(estimator, DATASETPARAMS)
    train_ds = LGBM_DatasetCreateFromMat(X, ds_parameters, is_row_major)
    LGBM_DatasetSetField(train_ds, "label", y)
    if length(weights) > 0
        LGBM_DatasetSetField(train_ds, "weight", weights)
    end
    if length(init_score) > 0
        LGBM_DatasetSetField(train_ds, "init_score", init_score)
    end

    @debug "Started creating LGBM booster\n"
    bst_parameters = stringifyparams(estimator, BOOSTERPARAMS) * " verbosity=$verbosity"
    estimator.booster = LGBM_BoosterCreate(train_ds, bst_parameters)

    n_tests = length(test)
    tests_names = Array{String}(undef,n_tests)

    if n_tests > 0
        @debug "Started creating LGBM test datasets\n"
        @inbounds for (test_idx, test_entry) in enumerate(test)
            tests_names[test_idx] = "test_$(test_idx)"
            test_ds = LGBM_DatasetCreateFromMat(test_entry[1], ds_parameters, train_ds, is_row_major)
            LGBM_DatasetSetField(test_ds, "label", test_entry[2])
            LGBM_BoosterAddValidData(estimator.booster, test_ds)
        end
    end

    @debug "Started training...\n"
    results = train(estimator, tests_names, verbosity, start_time)
    # estimator.model = readlines("$(tempdir)/model.txt")

    return results
end

function train(estimator::LGBM, tests_names::Vector{String}, verbosity::Integer,
               start_time::DateTime)
    ghtresults = Dict{String,Dict{String,Vector{Float64}}}()
    n_tests = length(tests_names)
    metrics = LGBM_BoosterGetEvalNames(estimator.booster)
    n_metrics = length(metrics)
    bigger_is_better = [ifelse(in(metric, MAXIMIZE_METRICS), 1., -1.) for metric in metrics]
    best_score = fill(-Inf, (n_metrics, n_tests))
    best_iter = fill(1, (n_metrics, n_tests))

    for iter in 1:estimator.num_iterations
        is_finished = LGBM_BoosterUpdateOneIter(estimator.booster)
        @debug(Dates.CompoundPeriod(now() - start_time),
                  " elapsed, finished iteration ", iter, "\n")
        if is_finished == 0
            is_finished = eval_metrics!(results, estimator, tests_names, iter, n_metrics,
                                        verbosity, bigger_is_better, best_score, best_iter, metrics)
        else
            shrinkresults!(results, iter - 1)
            @info("Stopped training because there are no more leaves that meet the ",
                     "split requirements.")
        end
        is_finished == 1 && return results
    end
    return results
end

function eval_metrics!(results::Dict{String,Dict{String,Vector{Float64}}},
                       estimator::LGBM, tests_names::Vector{String}, iter::Integer,
                       n_metrics::Integer, verbosity::Integer,
                       bigger_is_better::Vector{Float64}, best_score::Matrix{Float64},
                       best_iter::Matrix{Int}, metrics::Vector{String})
    if (iter - 1) % estimator.metric_freq == 0
        if estimator.is_training_metric
            scores = LGBM_BoosterGetEval(estimator.booster, 0)
            store_scores!(results, estimator, iter, "training", scores, metrics)
            print_scores(estimator, iter, "training", n_metrics, scores, metrics, verbosity)
        end
    end

    if (iter - 1) % estimator.metric_freq == 0 || estimator.early_stopping_round > 0
        for (test_idx, test_name) in enumerate(tests_names)
            scores = LGBM_BoosterGetEval(estimator.booster, test_idx)

            # Check if progress should be stored and/or printed
            if (iter - 1) % estimator.metric_freq == 0
                store_scores!(results, estimator, iter, test_name, scores, metrics)
                print_scores(estimator, iter, test_name, n_metrics, scores, metrics, verbosity)
            end

            # Check if early stopping is called for
            @inbounds for metric_idx in eachindex(metrics)
                maximize_score = bigger_is_better[metric_idx] * scores[metric_idx]
                if maximize_score > best_score[metric_idx, test_idx]
                    best_score[metric_idx, test_idx] = maximize_score
                    best_iter[metric_idx, test_idx] = iter
                elseif iter - best_iter[metric_idx, test_idx] >= estimator.early_stopping_round
                    shrinkresults!(results, best_iter[metric_idx, test_idx])
                    @info("Early stopping at iteration ", iter,
                             ", the best iteration round is ", best_iter[metric_idx, test_idx], "\n")
                    return 1
                end
            end
        end
    end

    return 0
end

function store_scores!(results::Dict{String,Dict{String,Vector{Float64}}},
                       estimator::LGBM, iter::Integer, evalname::String,
                       scores::Vector{Cdouble}, metrics::Vector{String})
    for (metric_idx, metric_name) in enumerate(metrics)
        if !haskey(results, evalname)
            num_evals = cld(estimator.num_iterations, estimator.metric_freq)
            results[evalname] = Dict{String,Vector{Float64}}()
            results[evalname][metric_name] = Array{Float64}(undef,num_evals)
        elseif !haskey(results[evalname], metric_name)
            num_evals = cld(estimator.num_iterations, estimator.metric_freq)
            results[evalname][metric_name] = Array{Float64}(undef,num_evals)
        end
        eval_idx = cld(iter, estimator.metric_freq)
        results[evalname][metric_name][eval_idx] = scores[metric_idx]
    end

    return nothing
end

function print_scores(estimator::LGBM, iter::Integer, name::String, n_metrics::Integer,
                      scores::Vector{Cdouble}, metrics::Vector{String}, verbosity::Integer)
    @info("Iteration: ", iter, ", ", name, "'s ")
    for (metric_idx, metric_name) in enumerate(metrics)
        @info( metric_name, ": ", scores[metric_idx])
        metric_idx < n_metrics && @info(verbosity, ", ")
    end
    @info(verbosity, "\n")
end

function stringifyparams(estimator::LGBM, params::Vector{Symbol})
    paramstring = ""
    n_params = length(params)
    valid_names = fieldnames(typeof(estimator))
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
