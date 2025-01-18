#include "matrix.hpp"
#include "model.hpp"
#include "optimizers.hpp"
#include "activations.hpp"
#include "loss.hpp"
#include "data_loader.hpp"
#include "utils.hpp"
#include "evaluate.hpp"

#include <iostream>
#include <cassert>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>


int main()
{
    omp_set_num_threads(16);
    
    DataLoader loader;
    
    // Load data
    Matrix train_x = loader.load_from_csv("data/fashion_mnist_train_vectors.csv");
    Matrix train_y = loader.load_from_csv("data/fashion_mnist_train_labels.csv");
    Matrix test_x = loader.load_from_csv("data/fashion_mnist_test_vectors.csv");
    Matrix test_y = loader.load_from_csv("data/fashion_mnist_test_labels.csv");

    // Shuffle train data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<size_t> row_indices(train_x.rows());
    std::iota(row_indices.begin(), row_indices.end(), 0);
    std::shuffle(row_indices.begin(), row_indices.end(), gen);
    train_x = Matrix::shuffle(train_x, row_indices);
    train_y = Matrix::shuffle(train_y, row_indices);

    // Create a validation split
    Matrix val_x, val_y;
    std::tie(train_x, val_x) = Matrix::split(train_x, 0.1);
    std::tie(train_y, val_y) = Matrix::split(train_y, 0.1);

    // Normalize the data
    train_x = Matrix::div(train_x, 255.0);
    val_x = Matrix::div(val_x, 255.0);
    test_x = Matrix::div(test_x, 255.0);

    // One hot encode the labels
    train_y = Matrix::one_hot_encoding(train_y, 10);
    
    // Split into batches
    std::vector<Matrix> train_x_batches = Matrix::batch(train_x, 128);
    std::vector<Matrix> train_y_batches = Matrix::batch(train_y, 128);

    // Define the model
    ReLU relu;
    LeakyReLU leaky;
    Linear lin;
    Sigmoid sig;
    FullyConnectedLayer layer1(train_x_batches[0].cols(), 256, leaky);
    FullyConnectedLayer layer2(256, 32, leaky);
    FullyConnectedLayer layer3(32, train_y_batches[0].cols(), lin);
    DropoutLayer dropout(0.15);

    Sequential model({
        layer1,
        dropout,
        layer2,
        layer3,
    });

    Adam optimizer(model.parameters(), 0.0005, 0.99, 0.999, 1e-8);
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Training loop
    for (int epoch = 0; epoch < 35; epoch++) {
        float loss_sum = 0;
        for (size_t batch = 0; batch < train_x_batches.size(); batch++) {
            optimizer.zero_grad();
            Matrix output = model.forward(train_x_batches[batch]);
            float loss = CategoricalCrossEntropy().compute_error(train_y_batches[batch], output);
            loss_sum += loss;
            Matrix loss_derivative = CategoricalCrossEntropy().compute_error_derivative(train_y_batches[batch], output);
            model.backward(loss_derivative);
            optimizer.step();
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        float valid_acc = accuracy(val_y,Matrix::rowwise_argmax(model.forward(val_x)));
        std::cout << "Epoch: " << epoch << " AVG Loss: " << loss_sum / train_x_batches.size() << " VAL ACC: " << valid_acc << " Time elapsed: " << std::chrono::duration_cast<std::chrono::seconds> (end - begin).count() << std::endl;
   }

    float acc = accuracy(test_y,Matrix::rowwise_argmax(model.forward(test_x)));
    std::cout << "Test Accuracy: " << acc << std::endl;

    Matrix output = model.forward(train_x);
    Matrix predictions = Matrix::rowwise_argmax(output);
    loader.write_to_csv(predictions, "train_predictions.csv");
    

    output = model.forward(test_x);
    predictions = Matrix::rowwise_argmax(output);
    loader.write_to_csv(predictions, "test_predictions.csv");

}



