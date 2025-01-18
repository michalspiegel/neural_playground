#include "tensor.hpp"

template <typename T>
void Tensor<T>::backward() {
    if (this.requires_grad): {
        this.backwardFn();
    }
    

}

/***********************************************
*                  Constructors                *
***********************************************/

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape, T value, bool requires_grad=false) {
    this.shape = shape;
    this.data = vector<T>(product<size_t>(shape), value);
    this.requires_grad = requires_grad;
}

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> shape, std::vector<T> data, bool requires_grad=false) {
    if (product<size_t>(shape) != data.size()) {
        throw std::runtime_error("Tried to initialize tensor with data incompatible with the given shape.")
    }
    this.shape = shape;
    this.data = data;
    this.requires_grad = requires_grad;
}

/***********************************************
*                   Operators                  *
***********************************************/



/***********************************************
*                     Other                    *
***********************************************/

template <typename T>
T product(const std::vector<T>& vec) {
    T prod = 1;
    for (const T& val : vec) {
        prod *= val;
    }
    return prod;
}