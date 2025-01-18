#include "activations.hpp"
#include <algorithm>
#include <cmath>

/************************************************
 *                     ReLU
 ************************************************/

ReLU::ReLU() {}

float ReLU::apply(float input) {
    return std::max((float) 0.0, input);
}

float ReLU::derivative(float input) {
    if (input > 0) {
        return 1;
    } else {
        return 0;
    }
}


/************************************************
 *                 Leaky ReLU
 ************************************************/

LeakyReLU::LeakyReLU() {
    this->negative_slope = 0.01;
}

LeakyReLU::LeakyReLU(float negative_slope) {
    this->negative_slope = negative_slope;
}

float LeakyReLU::apply(float input) {
    if (input >= 0) {
        return input;
    } else {
        return this->negative_slope * input;
    }
}

float LeakyReLU::derivative(float input) {
    if (input > 0) {
        return 1;
    } else {
        return this->negative_slope;
    }
}

/************************************************
 *                    Linear
 ************************************************/

Linear::Linear() {
    this->slope = 1;
    this->bias = 0;
}

Linear::Linear(float slope) {
    this->slope = slope;
    this->bias = 0;
}

Linear::Linear(float slope, float bias) {
    this->slope = slope;
    this->bias = bias;
}

float Linear::apply(float input) {
    return this->slope * input + this->bias;
}

float Linear::derivative(float input) {
    return this->slope;
}

/************************************************
 *                    Sigmoid
 ************************************************/

Sigmoid::Sigmoid() {}

float Sigmoid::apply(float input) {
    return 1 / (1 + std::exp(-input));
}

float Sigmoid::derivative(float input) {
    return input;
}