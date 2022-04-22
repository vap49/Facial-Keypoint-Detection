from Layer import Layer
from jax.numpy import numpy as np
class Reshape(Layer):
  def __init__(self, input_shape, output_shape):
    self.input_shape = input_shape
    self.output_shape = output_shape

  def forward(self, input):
    return np.reshape(input, self.output_shape)
  
  def backward(self, o_grad, lr):
    return np.reshape(o_grad, self.input_shape)
