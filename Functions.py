from Activation import Activation
from jax.numpy import numpy as np
from Layer import Layer
class Tanh(Activation):
  def __init__(self):
    
    def tanh(x):
      return np.tanh(x)
    
    def tanh_prime(x):
      return 1 - np.tanh(x) ** 2
    
    super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
  def __init__(self):
    def sigmoid(x):
      return 1 / (1 + np.exp(-x))
    
    def sigmoid_prime(x):
      s = sigmoid(x)

      return s * (1-s)

    super().__init__(sigmoid, sigmoid_prime)
class Softmax_2(Layer):
  def __init__(self, input_node, softmax_node):
    self.w = np.random.randn(input_node, softmax_node) / input_node
    self.b = np.zeros(softmax_node)

  def forward(self, image):
    self.orginialShape = image.shape
    image_mod = image.flatten()
    self.input_mod = image_mod

    o_val = np.dot(image_mod, self.w) + self.b
    
    self.o_val = o_val

    exp_out = np.exp(o_val)

    return exp_out / np.sum(exp_out, axis = 0)
  
  def backward(self, o_grad, lr):
    for i, grad in enumerate(o_grad):
      if grad == 0: continue
    
      transformation_eq = np.exp(self.out)
      S_total = np.sum(transformation_eq)

      dy_dZ = -transformation_eq[i]*transformation_eq / (S_total **2)
      dy_dZ[i] = transformation_eq[i] * (S_total - transformation_eq[i]) / (S_total ** 2)

      dz_dW = self.input_mod
      dz_dB = 1
      dz_dX = self.w

      dL_dz = grad * dy_dZ

      dL_dW = dz_dW[np.newaxis].T @ dL_dz[np.newaxis]
      dL_db = dL_dz * dz_dB
      dL_dX = dz_dX @ dL_dz

      self.w -= lr * dL_dW
      self.b -= lr * dL_db

      return dL_dX.reshape(self.originalShape)

class MaxPool(Layer):
  def __init__(self, filter_size):
    self.filter_size = filter_size

  def image_region(self, image):
    n_height = image.shape[0]
    n_width = image.shape[1]

    self.image = image
    
    for i in range(n_height):
      for j in range(n_width):
        image_patch = image[(i * self.filter_size) : ( i * self.filter_size + self.filter_size), (j * self.filter_size) : (j * self.filter_size) : ( j * self.filter_size + self.filter_size)]
        yield image_patch, i, j 
  
  def forward(self, image):
    h ,w ,num_filters = image.shape
    output = np.zeros((h // self.filter_size, w // self.filter_size, num_filters))

    for image_patch, i , j in self.image_region(image):
      output[i,j] = np.amax(image_patch, axis = (0,1))
    return output

  def backward(self, o_grad):
    o_grad_maxpool= np.zeros(self.image.shape)

    for image_patch, i , j in self.image_region(self.image):
      h, w, num_filters = image_patch.shape
      max_val = np.amax(image_patch, axis = (0,1))

      for il in range(h):
        for jl in range(w):
          for kl in range(num_filters):
            if image_patch(i,j,kl) == max_val:
              o_grad_maxpool[i * self.filter_size + il, j * self.filter_size + jl, kl] = o_grad[i,j,kl]
    
    return o_grad_maxpool