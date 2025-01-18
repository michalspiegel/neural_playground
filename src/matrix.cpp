#include "matrix.hpp"
#include <vector>

/***********************************************
 *                Constructors                 *
 ***********************************************/
Matrix::Matrix(size_t rows, size_t cols, float value) {
    this->data = std::vector<float>(rows*cols, value);
    this->shape = std::make_tuple(rows, cols);
}
Matrix::Matrix(size_t rows, size_t cols, std::vector<float> data) {
    if (rows*cols != data.size()) {
        throw std::runtime_error("Tried to load vector into matrix with incompatible size.");
    }
    this->data = data;
    this->shape = std::make_tuple(rows, cols);
}

Matrix::Matrix(std::tuple<size_t, size_t> shape, float value) {
    size_t rows, cols;
    std::tie(rows, cols) = shape;
    this->data = std::vector<float>(rows * cols, value);
    this->shape = shape;
}
Matrix::Matrix(std::tuple<size_t, size_t> shape, std::vector<float> data) {
    size_t rows, cols;
    std::tie(rows, cols) = shape;
    if (rows*cols != data.size()) {
        throw std::runtime_error("Tried to load vector into matrix with incompatible size.");
    }
    this->data = data;
    this->shape = shape;
}

/***********************************************
 *              Getters & Setters              *
 ***********************************************/

float& Matrix::operator[](size_t row, size_t col) {
    if (row >= this->rows() || col >= this->cols()) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    if (this->transposed) {
        return data[col * this->rows() +  row];
    } 
    return data[row * this->cols() + col];
}

float Matrix::get(size_t row, size_t col) {
    if (row >= this->rows() || col >= this->cols()) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    if (transposed) {
        return data[col * this->rows() + row];
    }
    return this->data[row * this->cols() + col];
}
void Matrix::set(size_t row, size_t col, float value) {
    if (row >= this->rows() || col >= this->cols()) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    if (transposed) {
        this->data[col * this->rows() + row] = value;
    } else {
        this->data[row * this->cols() + col] = value;
    }
}

void Matrix::set(std::vector<float> data) {
    if (this->data.size() != data.size()) {
        throw std::runtime_error("Tried to load vector into matrix with incompatible size.");
    }
    this->data = data;
}

void Matrix::set_all(float value) {
    for (size_t i = 0; i < this->data.size(); i++) {
        this->data[i] = value;
    }
}

size_t Matrix::rows() const {
    return std::get<0>(this->shape);
}

size_t Matrix::cols() const {
    return std::get<1>(this->shape);
}

/***********************************************
 *                 Other                       *
 ***********************************************/


bool Matrix::isEqual(const Matrix& A) const {
    if (this->cols() != A.cols() ||
        this->rows() != A.rows()) {
        return false;
    }

    if (this->data != A.data) {
        return false;
    }

    if (this->transposed != A.transposed) {
        return false;
    }

    return true;


}

bool Matrix::operator==(const Matrix& A) const
{
    return isEqual(A);
}