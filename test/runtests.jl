using Test
# using Revise

using LiteGBM
using LiteGBM: Regression
using LiteGBM: Dataset, train!, save, load

function regression_data(n=10000)
    y = rand(n)
    X = y .* 5 .+  6 .+ randn(n)
    X = reshape(X, (n,1))
    X, y
end

function setup()
    X_train, y_train = regression_data()
    X_test, y_test = regression_data()
    train_set = Dataset(X_train, y_train)
    test_set = Dataset(X_test, y_test, train_set)
    train_set, test_set
end

@testset "datatset" begin

    train_set, test_set = setup()
    @test size(train_set) == (10000, 1)
    # @test test_set.reference == train_set.reference
end

@testset "training" begin

    train_set, test_set = setup()
    model = LiteGBM.LGBMRegression() 

    @test model.booster.handle == C_NULL

    LiteGBM.train!(model, train_set)
    LiteGBM.train!(model, train_set)
    LiteGBM.train!(model, train_set)

    @test LiteGBM.get_eval(model, :train) isa Dict
    @test LiteGBM.get_current_iter(model) == 3

    LiteGBM.add_test_data!(model, test_set)
    @test LiteGBM.get_eval(model, :valid) isa Dict

    @test length(LiteGBM.feature_importance(model)) == 1

end 

@testset "wrong order" begin

    train_set, test_set = setup()
    model = LiteGBM.LGBMRegression() 
    @test_throws ErrorException LiteGBM.add_test_data!(model, test_set)
    
end


@testset "save and loading" begin

    train_set, test_set = setup()
    model = LiteGBM.LGBMRegression() 
    @test_throws LiteGBM.BoosterNotTrained save(model, "model.txt")
    train!(model, train_set)
    save(model, "model.txt")
    @test isfile("model.txt")
    rm("model.txt")
end

# TODO: these can cause a segfault if the model doesn't have a valid handler!

# LiteGBM.get_eval_counts(model)
# LiteGBM.get_eval_names(model)
# LiteGBM.get_eval(model, :train)

# Snipped of code to make a unpack macro
# function train!(model :: LGBM{Regression, Inactive} , dataset) 
#     fields = fieldnames(LGBM)
#     values = tuple([getfield(model[1], k) for k in fields]...)
#     kwargs = NamedTuple{fields}(values)
#     model[1] = LGBM{Regression, Active}(kwargs...) # doesn't mutate it!
#     create_booster!(model[1], dataset)
# end