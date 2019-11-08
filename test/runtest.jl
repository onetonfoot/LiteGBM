using Test
using Revise

using LiteGBM
using LiteGBM: regression


# Create dataset test

function regression_data(n=10000)

    y = rand(n)
    X = y .* 5 .+  6 .+ randn(n)
    X = reshape(X, (n,1))
    X, y
end


X, y = regression_data()

model = LiteGBM.LGBM(regression)

dataset = LiteGBM.create_train_dataset(model, X, y)
size(dataset) == (10000, 1)

# dataset = LiteGBM.create_train_dataset(model, X, y, weights=rand(10000))

# Create Boostetr

LiteGBM.create_booster!(model, dataset)
@test model.booster.handle != C_NULL

# Traning test

LiteGBM.train!(model, dataset)
LiteGBM.train!(model, dataset)
LiteGBM.train!(model, dataset)

@test LiteGBM.get_current_iter(model) == 3

# Validation test

model = LiteGBM.LGBM(regression)
dataset = LiteGBM.create_train_dataset(model, X, y)
LiteGBM.create_booster!(model, dataset)


Xtest, ytest = regression_data(100)

valid_dataset = LiteGBM.create_valid_dataset(model, Xtest, ytest) # CrashES!!!! why

let
    model = LiteGBM.LGBM(regression)
    dataset = LiteGBM.create_train_dataset(model, X, y)
    LiteGBM.create_booster!(model, dataset)
    valid_dataset = LiteGBM.create_valid_dataset(model, Xtest, ytest)
end




Xtest, ytest = regression_data()
num_total_rows = 20000
new_ds_handle = Ref{Ptr{Nothing}}()

# LiteGBM.LibLightGBM.LGBM_DatasetCreateByReference(dataset.handle, num_total_rows, new_ds_handle)

valid_dataset = LiteGBM.create_valid_dataset(model, Xtest, ytest)



# Test Eval

LiteGBM.get_eval_counts(model)
LiteGBM.get_eval_names(model)
LiteGBM.get_eval(model, :train)
LiteGBM.get_eval(model, :valid)

