# NN-EXP-4-Implementation-of-MLP-for-Non-Linear-Separable-Problem
**AIM:**

To implement a perceptron for classification using Python

**EQUIPMENTS REQUIRED:**
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

**RELATED THEORETICAL CONCEPT:**
Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows
XOR truth table
![Img1](https://user-images.githubusercontent.com/112920679/195774720-35c2ed9d-d484-4485-b608-d809931a28f5.gif)

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below

![Img2](https://user-images.githubusercontent.com/112920679/195774898-b0c5886b-3d58-4377-b52f-73148a3fe54d.gif)

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.To separate the two outputs using linear equation(s), it is required to draw two separate lines as shown in figure below:
![Img 3](https://user-images.githubusercontent.com/112920679/195775012-74683270-561b-4a3a-ac62-cf5ddfcf49ca.gif)
For a problem resembling the outputs of XOR, it was impossible for the machine to set up an equation for good outputs. This is what led to the birth of the concept of hidden layers which are extensively used in Artificial Neural Networks. The solution to the XOR problem lies in multidimensional analysis. We plug in numerous inputs in various layers of interpretation and processing, to generate the optimum outputs.
The inner layers for deeper processing of the inputs are known as hidden layers. The hidden layers are not dependent on any other layers. This architecture is known as Multilayer Perceptron (MLP).
![Img 4](https://user-images.githubusercontent.com/112920679/195775183-1f64fe3d-a60e-4998-b4f5-abce9534689d.gif)
The number of layers in MLP is not fixed and thus can have any number of hidden layers for processing. In the case of MLP, the weights are defined for each hidden layer, which transfers the signal to the next proceeding layer.Using the MLP approach lets us dive into more than two dimensions, which in turn lets us separate the outputs of XOR using multidimensional equations.Each hidden unit invokes an activation function, to range down their output values to 0 or The MLP approach also lies in the class of feed-forward Artificial Neural Network, and thus can only communicate in one direction. MLP solves the XOR problem efficiently by visualizing the data points in multi-dimensions and thus constructing an n-variable equation to fit in the output values using back propagation algorithm

**Algorithm :**

Step 1 : Initialize the input patterns for XOR Gate
Step 2: Initialize the desired output of the XOR Gate
Step 3: Initialize the weights for the 2 layer MLP with 2 Hidden neuron 
              and 1 output neuron
Step 3: Repeat the  iteration  until the losses become constant and 
              minimum
              (i)  Compute the output using forward pass output
              (ii) Compute the error  
		          (iii) Compute the change in weight ‘dw’ by using backward 
                     propagation algorithm.
             (iv) Modify the weight as per delta rule.
             (v)   Append the losses in a list
Step 4 : Test for the XOR patterns.

** PROGRAM** 
```
import numpy as np

# Step 1: Initialize input patterns for XOR Gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Step 2: Initialize the desired output of the XOR Gate
Y = np.array([0, 1, 1, 0])

# Step 3: Initialize the weights for the 2-layer MLP with 2 Hidden neurons and 1 output neuron
input_size = 2
hidden_size = 2
output_size = 1

# Initialize weights randomly with small values
weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)

# Learning rate for weight updates
learning_rate = 0.1

# Number of training iterations
num_epochs = 10000

losses = []

# Step 3: Training the MLP
for epoch in range(num_epochs):
    total_error = 0

    for i in range(len(X)):
        # (i) Compute the output using forward pass
        input_layer = X[i]
        hidden_layer_input = np.dot(input_layer, weights_input_hidden)
        hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        output_layer_output = 1 / (1 + np.exp(-output_layer_input))

        # (ii) Compute the error
        error = Y[i] - output_layer_output
        total_error += error ** 2

        # (iii) Compute the change in weights 'dw' using backward propagation
        delta_output = error * output_layer_output * (1 - output_layer_output)
        delta_hidden = delta_output.dot(weights_hidden_output.T) * \
            hidden_layer_output * (1 - hidden_layer_output)

        # (iv) Modify the weights using the delta rule
        weights_hidden_output += hidden_layer_output.reshape(-1, 1) * delta_output * learning_rate
        weights_input_hidden += input_layer.reshape(-1, 1) * delta_hidden * learning_rate

    # (v) Append the losses in a list
    losses.append(total_error)

# Step 4: Testing the XOR patterns
for i in range(len(X)):
    input_layer = X[i]
    hidden_layer_input = np.dot(input_layer, weights_input_hidden)
    hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = 1 / (1 + np.exp(-output_layer_input))

    print(f"Input: {input_layer}, Predicted Output: {predicted_output}")

# Print the final weights
print("Final weights (input to hidden):\n", weights_input_hidden)
print("Final weights (hidden to output):\n", weights_hidden_output)
```
 **OUTPUT** 
Input: [0 0], Predicted Output: [0.258014]
Input: [0 1], Predicted Output: [0.69172371]
Input: [1 0], Predicted Output: [0.68877368]
Input: [1 1], Predicted Output: [0.38875897]
Final weights (input to hidden):
 [[0.76821745 5.19601006]
 [0.75800259 5.03647995]]
Final weights (hidden to output):
 [[-9.2992654 ]
 [ 7.18663238]]
** RESULT**
Thus, the Perceptron for classification using Python has been implementation and successfully executed.
