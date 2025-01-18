#include "optimizers.hpp"
#include "utils.hpp"

void Optimizer::zero_grad() {
    for (size_t i = 0; i < parameters.size(); i++) {
        parameters[i]->grad.set_all(0);
    }
}

/**************************************
                  SGD                 
 **************************************/

SGD::SGD(std::vector<std::shared_ptr<Parameter>> parameters, float learning_rate): Optimizer(parameters), learning_rate(learning_rate) {};

void SGD::step() {
    for (size_t i = 0; i < parameters.size(); i++) {
        parameters[i]->data = Matrix::add(parameters[i]->data, Matrix::mul(parameters[i]->grad, -learning_rate));
    }
}


/**************************************
           SGD with momentum                 
 **************************************/

SGDWithMomentum::SGDWithMomentum(std::vector<std::shared_ptr<Parameter>> parameters, float learning_rate, float momentum): Optimizer(parameters), learning_rate(learning_rate), momentum(momentum) {
    for (size_t i = 0; i < parameters.size(); i++) {
        velocities.push_back(Matrix(parameters[i]->data.shape, 0));
    }
}

void SGDWithMomentum::step() {
    for (size_t i = 0; i < parameters.size(); i++) {
        velocities[i] = Matrix::add(Matrix::mul(velocities[i], momentum), Matrix::mul(parameters[i]->grad, -learning_rate));
        parameters[i]->data = Matrix::add(parameters[i]->data, velocities[i]);
    }
}

/**************************************
                  AdaGrad                 
 **************************************/

AdaGrad::AdaGrad(std::vector<std::shared_ptr<Parameter>> parameters, float learning_rate, float epsilon): Optimizer(parameters), learning_rate(learning_rate), epsilon(epsilon) {
    for (size_t i = 0; i < parameters.size(); i++) {
        squared_gradients.push_back(Matrix(parameters[i]->data.shape, 0));
    }
}

void AdaGrad::step() {
    for (size_t i = 0; i < parameters.size(); i++) {
        squared_gradients[i] = Matrix::add(squared_gradients[i], Matrix::mul(parameters[i]->grad, parameters[i]->grad));
        Matrix adaptive_learning_rate = Matrix::mul(Matrix::add(Matrix::sqrt(squared_gradients[i]), epsilon), -learning_rate); 
        parameters[i]->data = Matrix::add(parameters[i]->data, Matrix::mul(parameters[i]->grad, adaptive_learning_rate));
    }
}

/**************************************
                  RMSprop                 
 **************************************/

RMSprop::RMSprop(std::vector<std::shared_ptr<Parameter>> parameters, float learning_rate, float decay, float epsilon): Optimizer(parameters), learning_rate(learning_rate), decay(decay), epsilon(epsilon) {
    for (size_t i = 0; i < parameters.size(); i++) {
        v.push_back(Matrix(parameters[i]->data.shape, 0));
    }
}

void RMSprop::step() {
    for (size_t i = 0; i < parameters.size(); i++) {
        v[i] = Matrix::add(Matrix::mul(v[i], decay), Matrix::mul(Matrix::mul(parameters[i]->grad, parameters[i]->grad), 1 - decay));
        Matrix adaptive_learning_rate = Matrix::mul(Matrix::add(Matrix::sqrt(v[i]), epsilon), -learning_rate);
        parameters[i]->data = Matrix::add(parameters[i]->data, Matrix::mul(parameters[i]->grad, adaptive_learning_rate)); 
    }
}


/**************************************
                  Adam                 
 **************************************/

 Adam::Adam(std::vector<std::shared_ptr<Parameter>> parameters, float learning_rate, float beta1, float beta2, float epsilon): Optimizer(parameters), learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon) {
    for (size_t i = 0; i < parameters.size(); i++) {
        v.push_back(Matrix(parameters[i]->data.shape, 0));
        m.push_back(Matrix(parameters[i]->data.shape, 0));
    }
 }

 void Adam::step() {
    for (size_t i = 0; i < parameters.size(); i++) {
        m[i] = Matrix::add(Matrix::mul(m[i], beta1), Matrix::mul(parameters[i]->grad, 1 - beta1));
        v[i] = Matrix::add(Matrix::mul(v[i], beta2), Matrix::mul(Matrix::mul(parameters[i]->grad, parameters[i]->grad), 1 - beta2));
        Matrix velocity = Matrix::div(m[i], 1.0f - std::pow(beta1, t+1));
        Matrix scaled_v = Matrix::div(v[i], 1.0f - std::pow(beta2, t+1));
        Matrix adaptive_learning_rate = Matrix::div(-learning_rate, Matrix::add(Matrix::sqrt(scaled_v), epsilon));
        parameters[i] ->data = Matrix::add(parameters[i]->data, Matrix::mul(adaptive_learning_rate, velocity));
        t++;
    }
 }


/**************************************
                  AdamW                 
 **************************************/

void AdamW::step() {
    for (size_t i = 0; i < parameters.size(); ++i) {
        parameters[i]->data = Matrix::mul(parameters[i]->data, 1 - weight_decay);
    }
    Adam::step();
}