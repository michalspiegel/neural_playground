#include "tensor.hpp"

#include <algorithm>
#include <iterator>
#include <queue>
#include <optional>
 

/***********************************************
*                    Helpers                   *
***********************************************/

template <typename T>
T vector_product(std::vector<T>& vec) {
    T prod = 1;
    for ( T& val : vec) {
        prod *= val;
    }
    return prod;
}

template <typename T>
std::vector<T> mul_vectors(std::vector<T>& A,  std::vector<T>& B, 
                           std::vector<T>* output = nullptr) {
    if (A.size() != B.size()) {
        throw std::runtime_error("Tried to multiply vectors of different size.");
    }

    std::vector<T> new_vector;
    std::vector<T>* result = output ? output : &new_vector;
    result->resize(A.size());

    for (size_t i = 0; i < A.size(); i++) {
        (*result)[i] = A[i] * B[i];
    }

    return *result;
}

template <typename T>
std::vector<T> add_vectors(std::vector<T>& A,  std::vector<T>& B,
                           std::vector<T>* output = nullptr) {
    if (A.size() != B.size()) {
        throw std::runtime_error("Tried to add vectors of different size.");
    }
  
    std::vector<T> new_vector;
    std::vector<T>* result = output ? output : &new_vector;
    result->resize(A.size());

    for (size_t i = 0; i < A.size(); i++) {
        (*result)[i] = A[i] + B[i];
    }

    return *result;
}

/***********************************************
*                  Backward                    *
***********************************************/

template <typename T>
void Tensor<T>::backward() {
    if (!this->requires_grad) {
        return;
    }
    this->grad = {1};
    std::queue<std::shared_ptr<Tensor<T>>> q;
    q.push(std::make_shared<Tensor<T>>(*this));
    while (!q.empty()) {
        std::shared_ptr<Tensor<T>> node = q.front();
        q.pop();
        node->backwardFn();
        for (std::shared_ptr<Tensor<T>> child: node->prev) {
            q.push(child);
        }
    }
}

/***********************************************
*                  Constructors                *
***********************************************/

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape, T value, bool requires_grad) {
    this->shape = shape;
    this->data = std::vector<T>(vector_product<size_t>(shape), value);
    this->requires_grad = requires_grad;
    this->backwardFn = []{};
    this->prev = {};
    this->op = std::string("-");
    this->grad = std::vector<T>(data.size(), 0);
}

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape, std::vector<T> data, bool requires_grad) {
    if (vector_product<size_t>(shape) != data.size()) {
        throw std::runtime_error("Tried to initialize tensor with data incompatible with the given shape.");
    }
    this->shape = shape;
    this->data = data;
    this->requires_grad = requires_grad;
    this->backwardFn = []{};
    this->prev = {};
    this->op = std::string("-");
    this->grad = std::vector<T>(data.size(), 0);
}

/***********************************************
*                   Operators                  *
***********************************************/

template <typename T>
Tensor<T> Tensor<T>::operator+(Tensor& other) {
    if (this->shape != other.shape) {
        throw std::runtime_error("Tried to add tensors with incompatible shapes.");
    }

    bool requires_grad = this->requires_grad || other.requires_grad;
    Tensor<T> result = Tensor<T>(this->shape, add_vectors(this->data, other.data), requires_grad);
    result.op = std::string("+");
    result.prev = {std::make_shared<Tensor<T>>(*this), std::make_shared<Tensor<T>>(other)};
    result.backwardFn = [&]{
        add_vectors(this->grad, result.grad, &this->grad);
        add_vectors(other.grad, result.grad, &other.grad);
    };
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(T other) {    
    bool requires_grad = this->requires_grad || other.requires_grad;
    Tensor<T> result = Tensor<T>(this->shape, , requires_grad);
    result.op = std::string("+");
    result.prev = {std::make_shared<Tensor<T>>(*this), std::make_shared<Tensor<T>>(other)};
    result.backwardFn = [&]{
        add_vectors(this->grad, result.grad, &this->grad);
        add_vectors(other.grad, result.grad, &other.grad);
    };
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(Tensor& other)  {
    if (this->shape != other.shape) {
        throw std::runtime_error("Tried to multiply tensors with incompatible shapes.");
    }

    bool requires_grad = this->requires_grad || other.requires_grad;
    Tensor<T> result = Tensor<T>(this->shape, mul_vectors(this->data, other.data), requires_grad);
    result.op = std::string("*");
    result.prev = {std::make_shared<Tensor<T>>(*this), std::make_shared<Tensor<T>>(other)};
    result.backwardFn = [&]{
        std::vector<T> chain1 = mul_vectors(result.grad, other.data);
        add_vectors(this->grad, chain1, &this->grad);
        std::vector<T> chain2 = mul_vectors(result.grad, this->data);
        add_vectors(other.grad, chain2, &other.grad);
    };
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator/(Tensor& other) {

}

template<typename T>
Tensor<T> Tensor<T>::operator/(Tensor& other) {
    
}

template<typename T>
Tensor<T> Tensor<T>::operator/(Tensor& other) {
    
}
