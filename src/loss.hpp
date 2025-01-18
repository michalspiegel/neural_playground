#pragma once
#include "matrix.hpp"
#include "model.hpp"
#include <vector>

class Loss {
    public:
        virtual float compute_error(Matrix ground_truth, Matrix predicted_values) = 0;
        virtual Matrix compute_error_derivative(Matrix ground_truth, Matrix predicted_values) = 0;
};

class BinaryCrossEntropy: public Loss {
    public:
        float compute_error(Matrix ground_truth, Matrix predicted_values) override;
        Matrix compute_error_derivative(Matrix ground_truth, Matrix predicted_values) override;
};

class CategoricalCrossEntropy: public Loss {
    public:
        float compute_error(Matrix ground_truth, Matrix predicted_values) override;
        Matrix compute_error_derivative(Matrix ground_truth, Matrix predicted_values) override;
};

class MeanSquaredError: public Loss {
    public:
        float compute_error(Matrix ground_truth, Matrix predicted_values) override;
        Matrix compute_error_derivative(Matrix ground_truth, Matrix predicted_values) override;
};
