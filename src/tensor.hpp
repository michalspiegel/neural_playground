#pragma once

#include <vector>
#include <iostream>
#include <functional>

/**
 * @brief This class represents tensor objects and implements automatic differentiation
 * 
 */
 template <typename T>
class Tensor {
    private:
        std::function<void()> backwardFn;
        std::vector<Tensor> children;
    public:
        std::vector<size_t> shape;
        std::vector<T> data;
        std::vector<T> grad;
        bool requires_grad;

        void backward();

        /***********************************************
         *                Constructors                 *
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
        Tensor operator+(const Tensor& other) const;
        
        /**
         * @brief Piecewise division
         * 
         * @param other 
         * @return Tensor 
         */
        Tensor operator/(const Tensor& other) const;
        
        /**
         * @brief Piecewise substraction
         * 
         * @param other 
         * @return Tensor 
         */
        Tensor operator-(const Tensor& other) const;
        
        /**
         * @brief Piecewise multiplication
         * 
         * @param other 
         * @return Tensor 
         */
        Tensor operator*(const Tensor& other) const;
        
        /**
         * @brief Matrix multiplication
         * 
         * @param other 
         * @return Tensor 
         */
        Tensor operator%(const Tensor& other) const;
        
        /***********************************************
         *                    Other                    *
         ***********************************************/
        
        /**
         * @brief Transpose this tensor
         * 
         * @return Tensor 
         */
        Tensor transpose();
}