#pragma once

#include <cstddef>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <functional>
#include <string>
#include <iostream>
#include <cmath>
#include <omp.h>
#include <random>


/**
 * @brief Class for efficient matrix representation and operations.
 * TODO: Currently only supports float, future support for other is planned
 */
class Matrix {
    private:
        bool transposed=false;
        std::vector<float> data;
        bool isEqual(const Matrix& A) const;
    public:
        bool operator==(const Matrix& A) const;
        /**
         * @brief Stores the shape of the stored matrix
         * 
         */
        std::tuple<size_t, size_t> shape;
     
        /***********************************************
         *                Constructors                 *
         ***********************************************/
        Matrix() = default;
        Matrix(size_t rows, size_t cols, float value);
        Matrix(size_t rows, size_t cols, std::vector<float> data);
        Matrix(std::tuple<size_t, size_t> shape, float value);
        Matrix(std::tuple<size_t, size_t> shape, std::vector<float> data);
        
        /***********************************************
         *               Getters & Setters             *
         ***********************************************/
        
        float &operator[](size_t row, size_t col);

        /**
         * @brief Return the value at the given row and column
         *
         * @param row 
         * @param col 
         * @return float 
         */
        float get(size_t row, size_t col);
        
        /**
         * @brief Set the value at the given row and column
         *
         * @param row 
         * @param col 
         * @param value 
         */
        void set(size_t row, size_t col, float value);

        /**
         * @brief Set the underlying data of the matrix
         * 
         * @param data 
         */
        void set(std::vector<float> data);
        
        /**
         * @brief Set all elements of the matrix to the given value
         * 
         * @param value 
         */
        void set_all(float value);
        
        /**
         * @brief Return the number of rows in the matrix
         * 
         * @return size_t 
         */
        size_t rows() const;

        /**
         * @brief Return the number of columns in the matrix
         * 
         * @return size_t 
         */
        size_t cols() const;

        /***********************************************
         *               Matrix Operations             *
         ***********************************************/

        /**
            * @brief Transpose the matrix (switch rows for columns)
            * 
            * @return Matrix 
            */
        Matrix transpose() {
            Matrix transposed_m(this->cols(), this->rows(), this->data);
            transposed_m.transposed = true;
            return transposed_m;
        }

        /**
         * @brief Return a copy of the matrix
         * 
         * @return Matrix 
         */
        Matrix copy() {
            Matrix res(this->shape, this->data);
            if (this->transposed) {
                res.transposed = true;
            }
            return res;
        }

                

        /***********************************************
         *          Linear Algebra Operations          *
         ***********************************************/

        /**
         * @brief Perform matrix multiplication
         *
         * @param A 
         * @param B 
         * @return Matrix 
         */
        static Matrix matMul(Matrix A, Matrix B) {
            size_t A_row_count, A_col_count, B_row_count, B_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;
            std::tie(B_row_count, B_col_count) = B.shape;

            if (A_col_count != B_row_count) {
                throw std::runtime_error(std::string("Tried to multiply matrices with incompatible dimensions."));
            }

            Matrix result(A_row_count,  B_col_count, 0);
            // Switched inner loops to optimize CPU cache usage
            
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t i = 0; i < A_col_count; i++) {
                    for (size_t col = 0; col < B_col_count; col++) {
                        result[row, col] += A[row, i] * B[i, col];
                    }
                    
                }
            }
            return result;
        }

        /**
         * @brief Perform piecewise matrix addition
         *
         * @param A 
         * @param B 
         * @return Matrix 
         */
        static Matrix add(Matrix A, Matrix B) {
            size_t A_row_count, A_col_count, B_row_count, B_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;
            std::tie(B_row_count, B_col_count) = B.shape;

            if (A_row_count != B_row_count || A_col_count != B_col_count) {
                throw std::runtime_error(std::string("Tried to add matrices with incompatible dimensions."));
            }

            Matrix result(A_row_count, B_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < B_col_count; col++) {
                    result[row, col] = A[row, col] + B[row, col];
                }
            }
            return result;
        }

        static Matrix sub(Matrix A, Matrix B) {
            size_t A_row_count, A_col_count, B_row_count, B_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;
            std::tie(B_row_count, B_col_count) = B.shape;

            if (A_row_count != B_row_count || A_col_count != B_col_count) {
                throw std::runtime_error(std::string("Tried to add matrices with incompatible dimensions."));
            }

            Matrix result(A_row_count, B_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < B_col_count; col++) {
                    result[row, col] =  A[row, col] - B[row, col];
                }
            }
            return result;
        }



        /**
         * @brief Perform piecewise matrix multiplication
         *
         * @param A 
         * @param B 
         * @return Matrix 
         */
        static Matrix mul(Matrix A, Matrix B) {
            size_t A_row_count, A_col_count, B_row_count, B_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;
            std::tie(B_row_count, B_col_count) = B.shape;

            if (A_row_count != B_row_count || A_col_count != B_col_count) {
                throw std::runtime_error(std::string("Tried to multiply matrices with incompatible dimensions."));
            }

            Matrix result(A_row_count, B_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < B_col_count; col++) {
                    result[row, col] = A[row, col] * B[row, col];
                }
            }
            return result;
        }

        /**
         * @brief Perform piecewise matrix addition of a constant
         *
         * @param A 
         * @param value 
         * @return Matrix 
         */
        static Matrix add(Matrix A, float value) {
            size_t A_row_count, A_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;

            Matrix result(A_row_count, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < A_col_count; col++) {
                    result[row, col] = A[row, col] + value;
                }
            }
            return result;
        }

        static Matrix sub(Matrix A, float value) {
            size_t A_row_count, A_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;

            Matrix result(A_row_count, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < A_col_count; col++) {
                    result[row, col] = A[row, col] - value;
                }
            }
            return result;
        } 

        static Matrix sub(float value, Matrix A) {
            size_t A_row_count, A_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;

            Matrix result(A_row_count, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < A_col_count; col++) {
                    result[row, col] = value - A[row, col];
                }
            }
            return result;
        } 

        /**
         * @brief Perform piecewise matrix multiplication
         * 
         * @param A 
         * @param value 
         * @return Matrix 
         */
        static Matrix mul(Matrix A, float value) {
            size_t A_row_count, A_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;

            Matrix result(A_row_count, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < A_col_count; col++) {
                    result[row, col] = A[row, col] * value;
                }
            }
            return result;
        }

        /**
         * @brief Perform piecewise matrix multiplication
         * 
         * @param A 
         * @param B 
         * @return Matrix 
         */
        static Matrix div(Matrix A, float value) {
            size_t A_row_count, A_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;

            Matrix result(A_row_count, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < A_col_count; col++) {
                    result[row, col] = A[row, col] / value;
                }
            }
            return result;
        }

        static Matrix div(float value, Matrix A) {
            size_t A_row_count, A_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;

            Matrix result(A_row_count, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < A_col_count; col++) {
                    result[row, col] = value / A[row, col];
                }
            }
            return result;
        }
        
        /**
         * @brief Perform piecewise matrix division
         *
         * @param A 
         * @param B 
         * @return Matrix 
         */
        static Matrix div(Matrix A, Matrix B) {
            size_t A_row_count, A_col_count, B_row_count, B_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;
            std::tie(B_row_count, B_col_count) = B.shape;

            if (A_row_count != B_row_count || A_col_count != B_col_count) {
                throw std::runtime_error(std::string("Tried to divide matrices with incompatible dimensions."));
            }

            Matrix result(A_row_count, B_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < B_col_count; col++) {
                    result[row, col] = A[row, col] / B[row, col];
                }
            }
            return result;
        }

        /**
         * @brief Apply a function to each element of the matrix
         * 
         * @param A 
         * @param f 
         * @return Matrix 
         */
        static Matrix apply(Matrix A, std::function<float(float)> f) {
            size_t A_row_count, A_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;

            Matrix result(A_row_count, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < A_col_count; col++) {
                    result[row, col] = f(A[row, col]);
                }
            }
            return result;
        }

        /**
         * @brief Apply a function to each element of the matrices and perform piecewise multiplication
         * Allows to apply a function and add/multiply in one sweep efficiently
         * @param A 
         * @param B 
         * @param fA 
         * @param fB 
         * @return Matrix 
         */
        static Matrix apply_and_piecewise_mul(Matrix A, Matrix B, std::function<float(float)> fA, std::function<float(float)> fB) {
            size_t A_row_count, A_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;

            Matrix result(A_row_count, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < A_col_count; col++) {
                    result[row, col] = fA(A[row, col]) * fB(B[row, col]);
                }
            }
            return result;
        }

        /**
         * @brief Apply a function to each element of the matrices and perform piecewise addition
         * Allows to apply a function and add/multiply in one sweep efficiently
         * @param A 
         * @param B 
         * @param fA 
         * @param fB 
         * @return Matrix 
         */
        static Matrix apply_and_piecewise_add(Matrix A, Matrix B, std::function<float(float)> fA, std::function<float(float)> fB) {
            size_t A_row_count, A_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;

            Matrix result(A_row_count, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < A_col_count; col++) {
                    result[row, col] = fA(A[row, col]) + fB(B[row, col]);
                }
            }
            return result;
        }

        /**
         * @brief Sum the elements of each row of the matrix
         * 
         * @param A 
         * @return Matrix 
         */
        static Matrix rowwise_sum(Matrix A) {
            size_t A_row_count, A_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;

            Matrix result(1, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row = 0; row < A_row_count; row++) {
                float sum = 0;
                for (size_t col = 0; col < A_col_count; col++) {
                    sum += A[row, col];
                }
                result[row, 0] = sum;
            }
            return result;
        }
        
        /**
         * @brief Sum the elements of each column of the matrix
         * 
         * @param A 
         * @return Matrix 
         */
        static Matrix colwise_sum(Matrix A) {
            size_t A_row_count, A_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;

            Matrix result(1, A_col_count, 0);
            #pragma omp parallel for
            for (size_t col = 0; col < A_col_count; col++) {
                float sum = 0;
                for (size_t row = 0; row < A_row_count; row++) {
                    sum += A[row, col];
                }
                result[0, col] = sum;
            }
            return result;
        }

        static Matrix colwise_add(Matrix A, Matrix B) {
            size_t A_row_count, A_col_count, B_row_count, B_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;
            std::tie(B_row_count, B_col_count) = B.shape;

            if (B_row_count != 1 || A_col_count != B_col_count) {
                throw std::runtime_error(std::string("Tried to add matrices with incompatible dimensions. 2nd matrix must be a row vector."));
            }

            Matrix result(A_row_count, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < A_col_count; col++) {
                    result[row, col] = A[row, col] + B[0, col];
                }
            }
            return result;
        }

        static Matrix colwise_mul(Matrix A, Matrix B) {
            size_t A_row_count, A_col_count, B_row_count, B_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;
            std::tie(B_row_count, B_col_count) = B.shape;

            if (B_row_count != 1 || A_col_count != B_col_count) {
                throw std::runtime_error(std::string("Tried to add matrices with incompatible dimensions. 2nd matrix must be a row vector."));
            }

            Matrix result(A_row_count, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < A_col_count; col++) {
                    result[row, col] = A[row, col] * B[0, col];
                }
            }
            return result;
        }

        static Matrix colwise_sub(Matrix A, Matrix B) {
            size_t A_row_count, A_col_count, B_row_count, B_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;
            std::tie(B_row_count, B_col_count) = B.shape;

            if (B_row_count != 1 || A_col_count != B_col_count) {
                throw std::runtime_error(std::string("Tried to add matrices with incompatible dimensions. 2nd matrix must be a row vector."));
            }

            Matrix result(A_row_count, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < A_col_count; col++) {
                    result[row, col] = A[row, col] - B[0, col];
                }
            }
            return result;
        }

        static Matrix colwise_div(Matrix A, Matrix B) {
            size_t A_row_count, A_col_count, B_row_count, B_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;
            std::tie(B_row_count, B_col_count) = B.shape;

            if (B_row_count != 1 || A_col_count != B_col_count) {
                throw std::runtime_error(std::string("Tried to add matrices with incompatible dimensions. 2nd matrix must be a row vector."));
            }

            Matrix result(A_row_count, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < A_col_count; col++) {
                    result[row, col] = A[row, col] / B[0, col];
                }
            }
            return result;
        }

        static float sum(Matrix A) {
            float sum = 0;
            #pragma omp parallel for
            for (size_t row  = 0; row < A.rows(); row++) {
                for (size_t col = 0; col < A.cols(); col++) {
                    sum += A[row, col];
                }
            }
            return sum;
        }

        static Matrix clip(Matrix A, float lower, float upper) {
            size_t A_row_count, A_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;

            Matrix result(A_row_count, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < A_col_count; col++) {
                    if (A[row, col] > upper) {
                        result[row, col] = upper;
                    } else if (A[row, col] < lower) {
                        result[row, col] = lower;
                    } else {
                        result[row, col] = A[row, col];
                    }
                }
            }
            return result;
        }

        
        static std::vector<Matrix> batch(Matrix A, size_t batch_size, bool random = false) {
            std::vector<Matrix> batches;
            size_t num_batches = A.rows() / batch_size;

            for (size_t i = 0; i < num_batches; i++) {
                Matrix single_batch;
                if (A.data.begin() + (i+1) * batch_size * A.cols() > A.data.end()) {
                    std::vector<float> result(std::vector<float>(A.data.begin() + i * batch_size * A.cols(), A.data.end()));
                    single_batch = Matrix(result.size() / A.cols(), A.cols(), result);
                } else {
                    std::vector<float> result = std::vector<float>(A.data.begin() + i * batch_size * A.cols(), A.data.begin() + (i+1) * batch_size * A.cols());
                    single_batch = Matrix(batch_size, A.cols(), result);
                }
                batches.push_back(single_batch);
            }
            return batches;
        }

        static Matrix shuffle(Matrix A, std::vector<size_t> row_indices={}) {
            std::random_device rd;
            std::mt19937 gen(rd());

            if (row_indices.size() == 0) {
                row_indices = std::vector<size_t>(A.rows());
                std::iota(row_indices.begin(), row_indices.end(), 0);
                std::shuffle(row_indices.begin(), row_indices.end(), gen);
            }

            Matrix shuffled(A.rows(), A.cols(), 0);

            for (size_t new_row = 0; new_row < A.rows(); ++new_row) {
                size_t original_row = row_indices[new_row];
                std::copy(
                    A.data.begin() + original_row * A.cols(),
                    A.data.begin() + (original_row + 1) * A.cols(),
                    shuffled.data.begin() + new_row * A.cols()
                );
            }

            return shuffled;

        }

        
        static Matrix one_hot_encoding(Matrix labels, size_t num_classes) {
            Matrix result(labels.rows(), num_classes, 0);
            #pragma omp parallel for
            for (size_t i = 0; i < labels.rows(); i++) {
                result[i, labels[i, 0]] = 1;
            }
            return result;
        }

        static Matrix softmax(Matrix A) {
            
            // Compute exponentials of the inputs
            // Subtract the maximum value in each row to prevent overflow
            Matrix exp_inputs(A.rows(), A.cols(), 0);
            #pragma omp parallel for
            for (size_t i = 0; i < A.rows(); ++i) {
                float max_val = A[i, 0];
                for (size_t j = 0; j < A.cols(); ++j) {
                    if (A[i, j] > max_val) {
                        max_val = A[i, j];
                    }
                }
                for (size_t j = 0; j < A.cols(); ++j) {
                    exp_inputs[i, j] = std::exp(A[i,j] - max_val);
                }
            }

            // Sum the exponentials along the rows
            std::vector<float> row_sums(A.rows(), 0);
            #pragma omp parallel for
            for (size_t i = 0; i < A.rows(); ++i) {
                for (size_t j = 0; j < A.cols(); ++j) {
                    row_sums[i] += exp_inputs[i, j];
                }
            }

            // Divide each element in the exponentiated matrix by the corresponding row sum
            Matrix softmax_output(A.rows(), A.cols(), 0);
            #pragma omp parallel for
            for (size_t i = 0; i < A.rows(); ++i) {
                for (size_t j = 0; j < A.cols(); ++j) {
                    softmax_output[i, j] = exp_inputs[i, j] / row_sums[i];
                }
            }
            return softmax_output;
        }

        static Matrix rowwise_argmax(Matrix A) {
            Matrix result(A.rows(), 1, 0);
            #pragma omp parallel for
            for (size_t i = 0; i < A.rows(); i++) {
                float max_val = A[i, 0];
                size_t max_idx = 0;
                for (size_t j = 0; j < A.cols(); j++) {
                    if (A[i, j] > max_val) {
                        max_val = A[i, j];
                        max_idx = j;
                    }
                }
                result[i, 0] = max_idx;
            }
            return result;
        }

        static Matrix sqrt(Matrix A) {
            size_t A_row_count, A_col_count;
            std::tie(A_row_count, A_col_count) = A.shape;

            Matrix result(A_row_count, A_col_count, 0);
            #pragma omp parallel for
            for (size_t row  = 0; row < A_row_count; row++) {
                for (size_t col = 0; col < A_col_count; col++) {
                    result[row, col] = std::sqrt(A[row, col]);
                }
            }
            return result;
        }

        static std::tuple<Matrix, Matrix> split(Matrix A, float split_percentage) {
            if (split_percentage < 0.0f || split_percentage > 1.0f) {
                throw std::runtime_error("Split percentage must be between 0 and 1.");
            }

            size_t total_rows = A.rows();
            size_t split_index = static_cast<size_t>(total_rows * (1 - split_percentage));

            // Create the first split matrix
            Matrix first_split(split_index, A.cols(), 0);
            #pragma omp parallel for
            for (size_t i = 0; i < split_index; ++i) {
                for (size_t j = 0; j < A.cols(); ++j) {
                    first_split[i, j] = A[i, j];
                }
            }

            // Create the second split matrix
            Matrix second_split(total_rows - split_index, A.cols(), 0);
            #pragma omp parallel for
            for (size_t i = split_index; i < total_rows; ++i) {
                for (size_t j = 0; j < A.cols(); ++j) {
                    second_split[i - split_index, j] = A[i, j];
                }
            }

            return std::make_tuple(first_split, second_split);
        }
};

