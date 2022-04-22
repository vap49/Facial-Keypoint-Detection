class Layer:
  def __init__(self):
    self.input = None
    self.output = None

  def forward(self, input):
    pass
  
  def backward(self, o_grad, lr):
    """
      where o_grad = output gradient
            lr = learning rate
    """
    pass