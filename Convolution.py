from Layer import Layer
from jax.numpy import numpy as np
from scipy import signal
from jax._src.api import jit

class Convolutional(Layer):
  def __init__(self, input_shape, kernel_size, depth):
    self.input_depth, self.input_width, self.input_height = input_shape
    self.depth = depth
    self.input_shape = input_shape

    """
      y = x - K + 1
    """
    self.output_shape = (depth, self.input_height - kernel_size + 1, self.input_width - kernel_size + 1)
    self.kernels_shape = (depth, self.input_depth, kernel_size, kernel_size)
    self.kernels = np.random.normal(*self.kernels_shape)
    self.biases = np.random.normal(*self.output_shape)
  
  def forward(self, input):
    self.input = input
    self.output = np.copy(self.biases)
    
    """
      output(y) = Bias(b) + Sum( Input( x ) * Kernel( K ) )
      where: * = Cross Correlation
    
    """
    for i in range(self.depth):
      for j in range(self.input_depth):
        self.output[i] += signal.correlate2d(self.input[j], self.kernels[i,j], "valid")
    
    return self.output
  
  @jit
  def backward(self, o_grad, lr):
    #update params and return input grad

    kernels_gradient = np.zeros(self.kernels_shape)
    input_gradient = np.zeros(self.input_shape)

    for i in range(self.depth):
      for j in range(self.input_depth):
        kernels_gradient[ i , j ] = signal.correlate2d(self.input[j], o_grad[i], "valid")
        input_gradient[ j ] += signal.convolve2d(o_grad[i], self.kernels[ i , j ], "full")

    self.kernels -= lr * kernels_gradient
    self.biases -= lr * o_grad

    return input_gradient