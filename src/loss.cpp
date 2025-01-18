#include "loss.hpp"
#include "matrix.hpp"
#include <cmath>
#include <stdexcept>
#include "utils.hpp"

float CategoricalCrossEntropy::compute_error(Matrix ground_truth, Matrix predicted_values) {
    if (ground_truth.rows() != predicted_values.rows() || ground_truth.cols() != predicted_values.cols()) {
        throw std::runtime_error("Input matrices must have the same shape.");
    }
    Matrix softmaxed_preds = Matrix::softmax(predicted_values);
    float result = Matrix::sum(Matrix::apply_and_piecewise_mul(ground_truth, softmaxed_preds, [](float x) {return x;}, [](float x) {return std::log(x);}));
    return -result / ground_truth.rows();
}

Matrix CategoricalCrossEntropy::compute_error_derivative(Matrix ground_truth, Matrix predicted_values) {
    if (ground_truth.rows() != predicted_values.rows() || ground_truth.cols() != predicted_values.cols()) {
        throw std::runtime_error("Input matrices must have the same shape.");
    }
    Matrix softmaxed_preds = Matrix::softmax(predicted_values);
    return Matrix::sub(softmaxed_preds, ground_truth);
}

float MeanSquaredError::compute_error(Matrix ground_truth, Matrix predicted_values) {
    if (ground_truth.rows() != predicted_values.rows() || ground_truth.cols() != predicted_values.cols()) {
        throw std::runtime_error("Input matrices must have the same shape.");
    }
    float result = Matrix::sum(Matrix::apply(Matrix::apply_and_piecewise_add(ground_truth, predicted_values, [](float x) {return x;}, [](float x) {return -x;}), [](float x) {return x*x;}));
    return result / ground_truth.rows()*ground_truth.cols();
}

Matrix MeanSquaredError::compute_error_derivative(Matrix ground_truth, Matrix predicted_values) {
    if (ground_truth.rows() != predicted_values.rows() || ground_truth.cols() != predicted_values.cols()) {
        throw std::runtime_error("Input matrices must have the same shape.");
    }
    return Matrix::apply_and_piecewise_add(predicted_values, ground_truth, [](float x) {return x;}, [](float x) {return -1*x;});
}

float BinaryCrossEntropy::compute_error(Matrix ground_truth, Matrix predicted_values) {
    if (ground_truth.rows() != predicted_values.rows() || ground_truth.cols() != predicted_values.cols()) {
        throw std::runtime_error("Input matrices must have the same shape.");
    }
    Matrix clipped_preds = Matrix::clip(predicted_values, 1e-7, 1 - 1e-7);
    Matrix log_preds = Matrix::apply(clipped_preds, [](float x) { return std::log(x); });
    Matrix log_one_minus_preds = Matrix::apply(clipped_preds, [](float x) { return std::log(1.0f - x); });
    return -Matrix::sum(Matrix::add(Matrix::mul(ground_truth, log_preds), Matrix::mul(Matrix::sub(1.0f, ground_truth), log_one_minus_preds))) / ground_truth.rows();
}

Matrix BinaryCrossEntropy::compute_error_derivative(Matrix ground_truth, Matrix predicted_values) {
    if (ground_truth.rows() != predicted_values.rows() || ground_truth.cols() != predicted_values.cols()) {
        throw std::runtime_error("Input matrices must have the same shape.");
    }
    return Matrix::sub(predicted_values, ground_truth);
}

