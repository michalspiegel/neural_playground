#pragma once
#include "matrix.hpp"
#include "activations.hpp"

#include <memory>


/**
 * @brief A class to represent a parameter of a neural network model
 * 
 */
class Parameter {
    public:
        Matrix data;
        Matrix grad;
        
        Parameter() = default;
        Parameter(Matrix data) : data(data), grad(data.shape, 0) {}
};

/**
 * @brief A class to represent a neural network model
 * 
 */
class Model {
    public:
        virtual std::vector<std::shared_ptr<Parameter>> parameters() = 0;
        virtual Matrix forward(Matrix input, bool training=true) = 0;
        virtual Matrix backward(Matrix input) = 0;
};

/**
 * @brief  A class to represent a neural network model that is composed of a sequence of layers
 * 
 */
class Sequential: public Model { 
    public:
        Sequential(std::vector<std::reference_wrapper<Model>> layers);
        Matrix forward(Matrix input, bool training=true) override;
        Matrix backward(Matrix input) override;
        std::vector<std::shared_ptr<Parameter>> parameters() override;
    private:
        std::vector<std::reference_wrapper<Model>> layers;
};

/**
 * @brief  A class to represent a fully connected layer of a neural network model
 * 
 */
class FullyConnectedLayer : public Model {
    public:
        FullyConnectedLayer(size_t input_size, size_t output_size, std::reference_wrapper<ActivationFunction> activation_fn);
        Matrix forward(Matrix input, bool training=true) override;
        Matrix backward(Matrix output_gradient) override;
        std::vector<std::shared_ptr<Parameter>> parameters() override;
    private:
        std::shared_ptr<Parameter> weights;
        std::shared_ptr<Parameter> biases;
        Matrix inputs;
        Matrix inner_potential;
        std::reference_wrapper<ActivationFunction> activation_fn;
};

/**
 * @brief  A class to represent a dropout layer that randomly sets a fraction of input elements to zero
 * 
 */
class DropoutLayer : public Model {
    public:
        DropoutLayer(float dropout_rate);
        Matrix forward(Matrix input, bool training=true) override;
        Matrix backward(Matrix input) override;
        std::vector<std::shared_ptr<Parameter>> parameters() override;
    private:
        float dropout_rate;
        Matrix mask;
};

/**
 * @brief A class to represent a batch normalization layer of a neural network model
 * 
 */
class BatchNormLayer : public Model {
    public:
        BatchNormLayer(size_t size, float epsilon);
        Matrix forward(Matrix input, bool training=true) override;
        Matrix backward(Matrix) override;
        std::vector<std::shared_ptr<Parameter>> parameters() override;
    private:
        Matrix inputs;
        Matrix normalized_inputs;
        Matrix mean;
        Matrix std;
        size_t size;
        float epsilon;
        std::shared_ptr<Parameter> weights;
        std::shared_ptr<Parameter> biases;
};