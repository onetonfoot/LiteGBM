using Test
# using Revise

using LiteGBM
using LiteGBM: regression


function regression_data(n=10000)
    y = rand(n)
    X = y .* 5 .+  6 .+ randn(n)
    X = reshape(X, (n,1))
    X, y
end

@testset "training" begin

    X, y = regression_data()

    model = LiteGBM.LGBM(regression)
    @test model.booster.handle == C_NULL

    dataset = LiteGBM.create_train_dataset(model, X, y)
    @test size(dataset) == (10000, 1)

    LiteGBM.create_booster!(model, dataset)
    @test model.booster.handle != C_NULL

    Xtest, ytest = regression_data()
    valid_dataset = LiteGBM.create_valid_dataset(model, Xtest, ytest, dataset.handle)

    LiteGBM.train!(model, dataset)
    LiteGBM.train!(model, dataset)
    LiteGBM.train!(model, dataset)

    @test LiteGBM.get_current_iter(model) == 3
    @test LiteGBM.get_eval(model, :train) isa Dict
end 

# TODO: these can cause a segfault if the model doesn't have a valid handler!

# LiteGBM.get_eval_counts(model)
# LiteGBM.get_eval_names(model)
# LiteGBM.get_eval(model, :train)