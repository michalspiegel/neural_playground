#include "evaluate.hpp"

float accuracy(Matrix ground_truth, Matrix predicted) {
    if (ground_truth.rows() != predicted.rows() || ground_truth.cols() != 1 || predicted.cols() != 1) {
        throw std::runtime_error("Input matrices have incorrect shape");
    }
    float sum = 0;
    for (size_t i = 0; i < ground_truth.rows(); i++) {
        if (ground_truth[i, 0] == predicted[i, 0]) {
            sum++;
        }
    }
    return sum / ground_truth.rows();
}