This Jupyter Notebook implements a simple neural network classifier for the MNIST dataset. The MNIST dataset contains images of handwritten digits (0-9) and is commonly used for training various image processing systems.

## Dataset

The dataset used in this notebook is the MNIST dataset, which is loaded from a CSV file named `mnist_train.csv`. The dataset is split into a training set and a development set.

## Notebook Contents

1. **Data Loading and Preparation**:
  - The MNIST dataset is loaded using pandas and converted to a numpy array.
  - The data is shuffled and split into a training set and a development set.
  - The pixel values are normalized by dividing by 255.

2. **Initialization of Parameters**:
  - The weights and biases for the neural network are initialized randomly.

3. **Activation Functions**:
  - ReLU (Rectified Linear Unit) and Softmax activation functions are implemented.

4. **Forward Propagation**:
  - The forward propagation function computes the activations for each layer of the neural network.

5. **Backward Propagation**:
  - The backward propagation function computes the gradients for updating the weights and biases.

6. **Parameter Update**:
  - The parameters (weights and biases) are updated using gradient descent.

7. **Prediction and Accuracy**:
  - Functions to get predictions from the model and calculate the accuracy of the predictions.

8. **Training the Model**:
  - The model is trained using gradient descent for a specified number of iterations.

9. **Testing the Model**:
  - Functions to test the model on individual images and visualize the results.

10. **Development Set Evaluation**:
  - The model is evaluated on the development set to check its accuracy.

## Usage

To use this notebook, ensure you have the following dependencies installed:
- numpy
- pandas
- matplotlib

You can run the notebook cells sequentially to train the neural network and evaluate its performance on the MNIST dataset.

## Results

The notebook prints the accuracy of the model on the training set during training and on the development set after training. It also provides functions to visualize the predictions on individual images from the training set.
