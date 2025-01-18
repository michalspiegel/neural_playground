#include "data_loader.hpp"
#include <fstream>
#include <sstream>

std::ifstream open_file(const std::string filepath);
std::vector<float> parse_row(const std::string& line, size_t current_row);


Matrix DataLoader::load_from_csv(std::string filepath) {
    std::ifstream file = open_file(filepath);

    std::string line;
    std::vector<float> data;
    
    getline(file, line);
    size_t rows = 0;
    std::vector<float> row_data = parse_row(line, rows);
    size_t cols = row_data.size();
    do {
        row_data = parse_row(line, rows);
        if (cols != row_data.size()) {
            throw std::runtime_error("Error while loading data from CSV: Expected " + std::to_string(cols) + 
                                        "columns but encountered " + std::to_string(row_data.size()) + "at row " + 
                                        std::to_string(rows));
        }
        for (size_t col = 0; col < cols; ++col) {
            data.push_back(row_data[col]);
        }
        rows++;
    } while (getline(file, line));          

    Matrix matrix(rows, cols, 0);
    matrix.set(data);

    return matrix;
}

void DataLoader::write_to_csv(Matrix A, std::string filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    for (size_t row = 0; row < A.rows(); ++row) {
        for (size_t col = 0; col < A.cols(); ++col) {
            file << A[row, col];
            if (col < A.cols() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

std::ifstream open_file(const std::string filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    return file;
}

std::vector<float> parse_row(const std::string& line, size_t current_row) {
    std::stringstream line_stream(line);
    std::string cell;
    std::vector<float> row_data;

    while (getline(line_stream, cell, ',')) {
        try {
            float value = stof(cell);
            row_data.push_back(value);
        } catch (std::invalid_argument const& ex) {
            throw std::runtime_error("Invalid numerical value at row " + std::to_string(current_row) +
                                ", column " + std::to_string(row_data.size()));
        }
    }

    return row_data;
}
