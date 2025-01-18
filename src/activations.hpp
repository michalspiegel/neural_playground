#pragma once


/**
 * @brief Abstract class for neural network activation functions.
 * 
 * TODO: Currently supports only float values for computation, future support for more types is coming
 * 
 */
class ActivationFunction {
    public:
        virtual float apply(float input) = 0;
        virtual float derivative(float input) = 0;
};

/**
 * @brief Implementation of the ReLU activation function
 *
 */
class ReLU: public ActivationFunction {
    public:
        ReLU();
        float apply(float input) override;
        float derivative(float input) override;
};

/**
 * @brief Implementation of the LeakyReLU activation function
 * 
 */
class LeakyReLU: public ActivationFunction {
    public:
        /**
         * @brief Construct a new LeakyReLU object. 
         * 
         * The negative slope, which is used for negative input values, is set to 0.01 by default. 
         */
        LeakyReLU();
        /**
         * @brief Construct a new Leaky
         * 
         * @param negative_slope The negative slope is used for negative input values
         */
        LeakyReLU(float negative_slope);
        float apply(float input) override;
        float derivative(float input) override;
    
    private:
        float negative_slope;
};

/**
 * @brief Linear activation function
 * 
 */
class Linear: public ActivationFunction {
    public:
        Linear();
        Linear(float slope);
        Linear(float slope, float bias);
        float apply(float input) override;
        float derivative(float input) override;
    private:
        float slope;
        float bias;
};


class Sigmoid: public ActivationFunction {
    public:
        Sigmoid();
        float apply(float input) override;
        float derivative(float input) override;
};