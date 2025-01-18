//#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include <vector>

#include "model.hpp"
#include "matrix.hpp"
#include "utils.hpp"
#include "loss.hpp"


TEST_CASE("Test forward and backward on simple 1 layer network", "[model]") {
    
    ReLU relu;
    FullyConnectedLayer layer1(2, 1, relu);
    Sequential model({
        layer1
    });
    model.parameters()[0]->data.set({2, 4});
    model.parameters()[1]->data.set({-1});
    REQUIRE(model.forward(Matrix(2, 2, {-2, 1, 2, 3})) == Matrix(2, 1, {0, 15}));
    REQUIRE(MeanSquaredError().compute_error(Matrix(2, 1, {0, 0}), Matrix(2, 1, {0, 15})) == 112.5);
    model.backward(MeanSquaredError().compute_error_derivative(Matrix(2, 1, {0, 0}), Matrix(2, 1, {0, 15})));
    REQUIRE(model.parameters()[0]->grad == Matrix(2, 1, {30, 45}));
    REQUIRE(model.parameters()[1]->grad == Matrix(1, 1, {15}));
}

TEST_CASE("Test forward and backward on simple 2 layer network", "[model]") {
    
    ReLU relu;
    FullyConnectedLayer layer1(2, 2, relu);
    FullyConnectedLayer layer2(2, 1, relu);
    Sequential model({
        layer1,
        layer2
    });
    model.parameters()[0]->data.set({2, -2, 4, -4});
    model.parameters()[1]->data.set({-1, 15});
    model.parameters()[2]->data.set({4, -20});
    model.parameters()[3]->data.set({10});
    REQUIRE(model.forward(Matrix(2, 2, {-2, 1, 2, 3})) == Matrix(2, 1, {0, 70}));
    REQUIRE(MeanSquaredError().compute_error(Matrix(2, 1, {0, 0}), Matrix(2, 1, {0, 70})) == 2450);
    model.backward(MeanSquaredError().compute_error_derivative(Matrix(2, 1, {0, 0}), Matrix(2, 1, {0, 70})));
    REQUIRE(model.parameters()[0]->grad == Matrix(2, 2, {560, 0, 840, 0}));
    REQUIRE(model.parameters()[1]->grad == Matrix(1, 2, {280, 0}));
    REQUIRE(model.parameters()[2]->grad == Matrix(2, 1, {1050, 0}));
    REQUIRE(model.parameters()[3]->grad == Matrix(1, 1, {70}));
}