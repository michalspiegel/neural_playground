#include "utils.hpp"
#include "matrix.hpp"

#include <iostream>
#include <cmath>

void prettyPrintMatrix(Matrix input) {
    for (size_t row = 0; row < input.rows(); row++) {
        for (size_t col = 0; col < input.cols(); col++) {
            std::cout << input[row, col] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "END\n";
}

bool contains_nan(Matrix matrix) {
    for (size_t i = 0; i < matrix.rows(); ++i) {
        for (size_t j = 0; j < matrix.cols(); ++j) {
            if (std::isnan(matrix[i, j])) {
                return true;
            }
        }
    }
    return false;
}
