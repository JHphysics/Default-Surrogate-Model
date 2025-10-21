# Default-Surrogate-Model
The simplest surrogate model structure that can train from FEM results
This example was developed to illustrate the basic principles of a surrogate model and was implemented using a very simple DNN structure.
# Requirements : 
Since this is a very simple project, you just need a Python environment where you can run scripts and have TensorFlow installed. To accommodate environments without GPU support, the CPU version of TensorFlow has been installed.
(It also works with the GPU version of TensorFlow 😄)

I’m sharing the environment I set up. It’s not mandatory to have the exact same environment.

- Python : 3.8.19
- Tensorflow : 2.1.13
- scipy : 1.10.1
- matplotlib : 3.7.5

# Procedure: 

1. Load the data using MATLAB. (I used MATLAB for convenience, but other types of data can also be used.)

2. If you want to use data other than the example data, preprocess it into the shape (number of samples × number of classes).

3. Decide on the structure of the model. (In this example, 5 hidden layers with tanh activation were used.)

4. Check the results.
