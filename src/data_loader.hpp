#pragma once
#include "matrix.hpp"
#include <string>
#include <vector>

/**
 * @brief Class provides functionality for loading data into matrices
 * TODO: Currently only supports float, plans to support more data types in future
 */
class DataLoader {
    public:
        /**
        * @brief Load CSV numeric data from the specified file into a matrix. 
        * 
        * @param filepath 
        * @return Matrix 
        */
        Matrix load_from_csv(std::string filepath);

        void write_to_csv(Matrix A, std::string filepath);

};
