#pragma once
#include "model.hpp"
#include <vector>

class Optimizer {
    public:
        Optimizer(std::vector<std::shared_ptr<Parameter>> parameters): parameters(parameters) {};
        virtual void step() = 0;
        void zero_grad();
    protected:
        std::vector<std::shared_ptr<Parameter>> parameters;
};

class SGD: public Optimizer {
    public:
        SGD(std::vector<std::shared_ptr<Parameter>>, float learning_rate);
        void step() override;
    private:
        float learning_rate;
};


class SGDWithMomentum: public Optimizer {
    public:
        SGDWithMomentum(std::vector<std::shared_ptr<Parameter>>, float learning_rate, float momentum);
        void step() override;
    private:
        float learning_rate;
        float momentum;
        std::vector<Matrix> velocities;
};

class AdaGrad: public Optimizer {
    public:
        AdaGrad(std::vector<std::shared_ptr<Parameter>>, float learning_rate, float epsilon);
        void step() override;
    private:
        float learning_rate;
        float epsilon;
        std::vector<Matrix> squared_gradients;
};

class RMSprop: public Optimizer {
    public:
        RMSprop(std::vector<std::shared_ptr<Parameter>>, float learning_rate, float decay, float epsilon);
        void step() override;
    private:
        float learning_rate;
        float decay;
        float epsilon;
        std::vector<Matrix> v;
};

class Adam: public Optimizer {
    public:
        Adam(std::vector<std::shared_ptr<Parameter>>, float learning_rate, float beta1, float beta2, float epsilon);
        void step() override;
    private:
        float learning_rate;
        float beta1;
        float beta2;
        float epsilon;
        std::vector<Matrix> m;
        std::vector<Matrix> v;
        int t = 0;
}; 


class AdamW : public Adam {
public:
    AdamW(std::vector<std::shared_ptr<Parameter>> parameters, float learning_rate, float beta1, float beta2, float epsilon, float weight_decay)
        : Adam(parameters, learning_rate, beta1, beta2, epsilon), weight_decay(weight_decay) {}

    void step() override;

private:
    float weight_decay; // Specific to AdamW
};
class Overshoot: public Optimizer {
    public:
        Overshoot(std::vector<std::shared_ptr<Parameter>>, float learning_rate, float beta1, float beta2, float epsilon);
        void step() override;
    private:
        float learning_rate;
};

class SGDwithOvershoot: public Optimizer {
    public:
        SGDwithOvershoot(std::vector<std::shared_ptr<Parameter>>, float learning_rate, float beta1, float beta2, float epsilon);
        void step() override;
    private:
        float learning_rate;
};

class AdamwithOvershoot: public Optimizer {
    public:
        AdamwithOvershoot(std::vector<std::shared_ptr<Parameter>>, float learning_rate, float beta1, float beta2, float epsilon);
        void step() override;
    private:
        float learning_rate;
};