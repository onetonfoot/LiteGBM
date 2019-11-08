import .LibLightGBM



# train_set
# num_boost_round::Int
# valid_sets || valid_names # Can either pass a string to the dataset of the dataset
# fobj::Function # custom objective funciton
# feval::Function # custom eval func
# init_model::Union{String, Booster} # filename or booster object
# feature_name # list of strings or dataframe columns are used
# categorical_feature # normally auto # negaivte values will be treated as missing!
# early_stopping_rounds::Int # Will stop if validation score doens't improve after X rounds
# evals_results::Dict # used to store  eval results on the validation set
# verbose_eval::Union{Bool,Int} # 
# learning_rates::Union{Arrray{Real}, Funciton} #List of leanring rates for each boosting round or functino that take round and returns LR
# callbacks::Vector{Function} # List of functions to be called at end of each round