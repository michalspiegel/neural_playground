#include "model.hpp"
#include "utils.hpp"
#include <cmath>
#include <random>
#include <cassert>
#include <iostream>
#include <stdexcept>


/************************************************
 *              Fully Connected Layer           *
 ************************************************/

Matrix initialize_weights(size_t rows, size_t cols, float min, float max) {
    // Define the random number generator and distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    Matrix output(rows, cols, 0);
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            output[row, col] = dist(gen);
        }
    }
    return output;
}

FullyConnectedLayer::FullyConnectedLayer(size_t input_size, size_t output_size, std::reference_wrapper<ActivationFunction> activation_fn)
: weights(std::make_shared<Parameter>(Parameter(initialize_weights(input_size, output_size, -0.3, 0.3)))),
  biases(std::make_shared<Parameter>(Parameter(initialize_weights(1, output_size, -0.1, 0.1)))),
  activation_fn(std::move(activation_fn)) {}

Matrix FullyConnectedLayer::forward(Matrix input, bool training=true) {
    Matrix results = Matrix::matMul(input, weights->data);
    results = Matrix::colwise_add(results, biases->data);
    this->inner_potential = results.copy();
    results = Matrix::apply(results, [&](float x) {return activation_fn.get().apply(x);});
    this->inputs = input.copy();
    return results;
}    

Matrix FullyConnectedLayer::backward(Matrix loss_gradient) {
    loss_gradient = Matrix::mul(loss_gradient, Matrix::apply(inner_potential, [&](float x) {return activation_fn.get().derivative(x);}));
    weights->grad = Matrix::add(weights->grad, Matrix::matMul(inputs.transpose(), loss_gradient));
    biases->grad = Matrix::add(biases->grad, Matrix::colwise_sum(loss_gradient));
    Matrix next_loss_gradient = Matrix::matMul(loss_gradient, weights->data.transpose());
    return next_loss_gradient;
}

std::vector<std::shared_ptr<Parameter>> FullyConnectedLayer::parameters() {
    return {weights, biases};
}
/************************************************
 *                   Dropout                    *
 ************************************************/

DropoutLayer::DropoutLayer(float dropout_rate) : dropout_rate(dropout_rate) {}
Matrix DropoutLayer::forward(Matrix input, bool training=true) {

    if (!training) {
        return Matrix::mul(input, 1 - dropout_rate)
    }

    // Generate random mask to zero some inputs
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);
    Matrix output(input.shape, 0);
    mask = Matrix(input.shape, 1); 
    for (size_t row = 0; row < mask.rows(); row++) {
        for (size_t col = 0; col < mask.cols(); col++) {
            if (dist(gen) < dropout_rate) {
                mask[row, col] = 0;
            }
        }
    }
    return Matrix::mul(input, mask);
}

Matrix DropoutLayer::backward(Matrix input) {
    return Matrix::mul(input, mask);
}
std::vector<std::shared_ptr<Parameter>> DropoutLayer::parameters() {
    return {};
}

/************************************************
 *                  BatchNorm                   *
 ************************************************/

// NOT WORKING YET!!!

BatchNormLayer::BatchNormLayer(size_t size, float epsilon) : size(size), epsilon(epsilon), weights(std::make_shared<Parameter>(Parameter(Matrix(1, size, 1)))), biases(std::make_shared<Parameter>(Parameter(Matrix(1, size, 0)))) {}
Matrix BatchNormLayer::forward(Matrix input, bool training=true) {
    inputs = inputs.copy();
    mean = Matrix::div(Matrix::colwise_sum(input), input.rows());
    std = Matrix::sqrt(Matrix::add(Matrix::colwise_sum(Matrix::apply(Matrix::colwise_sub(input, mean), [](float x) {return x*x;})), epsilon));
    normalized_inputs = Matrix::colwise_div(Matrix::colwise_sub(input, mean), std);
    return Matrix::colwise_add(Matrix::colwise_mul(normalized_inputs, weights->data), biases->data);
}

// THIS BACKWARD IS NOT IMPLEMENTED YET

Matrix BatchNormLayer::backward(Matrix) {
    throw std::runtime_error(std::string("BatchNormLayer is not implemented yet!"));
}

std::vector<std::shared_ptr<Parameter>> BatchNormLayer::parameters() {
    return {weights, biases};
}

/************************************************
 *                  Sequential                  *
 ************************************************/

Sequential::Sequential(std::vector<std::reference_wrapper<Model>> layers) : layers(std::move(layers)) {}

Matrix Sequential::forward(Matrix input, bool training=true) {
    Matrix output = input;
    for (size_t i = 0; i < layers.size(); i++) {
        output = layers[i].get().forward(output, training);
    }
    return output;
}

Matrix Sequential::backward(Matrix input) {
    Matrix output = input;
    size_t i = layers.size();
    while (i > 0) {
        i--;
        output = layers[i].get().backward(output);        
    }
    return output;
}

std::vector<std::shared_ptr<Parameter>> Sequential::parameters() {
    std::vector<std::shared_ptr<Parameter>> params;
    for (size_t i = 0; i < layers.size(); i++) {
        std::vector<std::shared_ptr<Parameter>> layer_params = layers[i].get().parameters();
        params.reserve(params.size() + distance(layer_params.begin(),layer_params.end()));
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}