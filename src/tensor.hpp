#pragma once

#include <vector>
#include <iostream>
#include <functional>
#include <memory>
#include <string>

/**
 * @brief This class represents tensor objects and implements automatic differentiation
 * 
 */
 template <typename T>
class Tensor {
    private:
        std::function<void()> backwardFn;
        std::vector<std::shared_ptr<Tensor>> prev;
        std::string op;
    public:
        std::vector<size_t> shape;
        std::vector<T> data;
        std::vector<T> grad;
        bool requires_grad;

        void backward();

        /***********************************************
         *                ructors                 *
         ***********************************************/

        Tensor(std::vector<size_t> shape, T value, bool requires_grad=false);
        Tensor(std::vector<size_t> shape, std::vector<T> data, bool requires_grad=false);

        /***********************************************
         *                  Operators                  *
         ***********************************************/
        
        /**
         * @brief Piecewise addition
         * 
         * @param other 
         * @return Tensor 
         */
        Tensor operator+(Tensor& other);
        Tensor operator+(T other);

        
        /**
         * @brief Piecewise division
         * 
         * @param other 
         * @return Tensor 
         */
        Tensor operator/(Tensor& other);
        
        /**
         * @brief Piecewise substraction
         * 
         * @param other 
         * @return Tensor 
         */
        Tensor operator-(Tensor& other);
        
        /**
         * @brief Piecewise multiplication
         * 
         * @param other 
         * @return Tensor 
         */
        Tensor operator*(Tensor& other);
        
        /**
         * @brief Matrix multiplication
         * 
         * @param other 
         * @return Tensor 
         */
        Tensor operator%(Tensor& other);
        
        /***********************************************
         *                    Other                    *
         ***********************************************/
        
        /**
         * @brief Transpose this tensor
         * 
         * @return Tensor 
         */
        Tensor transpose();
};

template class Tensor<float>;
template class Tensor<double>;