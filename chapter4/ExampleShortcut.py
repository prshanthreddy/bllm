import torch
import torch.nn as nn
from GELU import GELU
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([       #1
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), 
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), 
                          GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)         #2
            if self.use_shortcut and x.shape == layer_output.shape:    #3
                x = x + layer_output
            else:
                x = layer_output
        return x

    def print_gradients(model, x):
        output = model(x)             #1
        target = torch.tensor([[0.]])

        loss = nn.MSELoss()
        loss = loss(output, target)    #2

        loss.backward()          #3

        for name, param in model.named_parameters():
            if 'weight' in name:
                print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


if __name__ == "__main__":
    x = torch.randn(2, 64)
    dnn = ExampleDeepNeuralNetwork([64, 128, 256, 512, 256, 64], True)
    print(dnn(x))
    print(dnn(x).shape)
    print(dnn(x).size())
    print(dnn(x).size(0))
    print(dnn(x).size(1))
    print(dnn.print_gradients(x))
    


#1 The layers are defined as a list of nn.Sequential instances, each containing a linear layer followed by a GELU activation function.
#2 The input x is passed through each layer in the list.
#3 If the use_shortcut flag is set to True and the output of the layer has the same shape as the input, a shortcut connection is added.
# The shortcut connection is implemented by adding the input x to the output of the layer. Otherwise, the output of the layer is used as the new input x.
# The final output of the network is returned after passing through all the layers.
# The ExampleDeepNeuralNetwork class defines a deep neural network with a configurable number of layers and shortcut connections.
# The network consists of a series of linear layers followed by GELU activation functions.
# The use_shortcut flag controls whether shortcut connections are added between layers.
# The forward method passes the input through each layer in the network and adds a shortcut connection if required.
# The output of the network is returned after passing through all the layers.
# The main block creates an instance of the ExampleDeepNeuralNetwork class and passes a random input tensor through the network.
# The output tensor is printed along with its shape and size.
# The size method is used to access the dimensions of the output tensor.
# The ExampleDeepNeuralNetwork class can be used to create deep neural networks with configurable layer sizes and shortcut connections.
# The network architecture consists of linear layers followed by GELU activation functions.
# Shortcut connections can be added between layers to improve training performance.
# The network can be used to process input data and produce output predictions.
# The network can be trained using gradient-based optimization algorithms to learn the underlying patterns in the data.
# The network can be evaluated on test data to assess its performance and generalization capabilities.
# The network can be deployed in production environments to make predictions on new data samples.
# The network can be fine-tuned on specific tasks or datasets to improve its performance and accuracy.
# The network can be used as a building block in larger machine learning systems or architectures.
# The network can be modified or extended to incorporate new features or capabilities as needed.
# The network can be optimized for performance or memory efficiency to run on resource-constrained devices.
# The network can be parallelized or distributed across multiple devices or machines to speed up training or inference.
# The network can be integrated with other machine learning models or algorithms to solve complex problems or tasks.
# The network can be adapted to different domains or applications by adjusting its architecture or hyperparameters.
# The network can be visualized or interpreted to understand its internal representations or decision-making process.
# The network can be analyzed or debugged to identify issues or improve its performance on specific tasks.


    