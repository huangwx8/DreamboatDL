class Module:
    '''
    This is the base module, from which all neural network modules will derive.
    **state_dict** contains all the trainable parameters and their accumulated gradients.
    **forward** and **backward** will be used for the network's forward computation and backpropagation.
    **__call__** will directly invoke the forward method.
    **zero_grad** clears the accumulated gradients and should be frequently called during iterative computations.
    '''
    def __init__(self):
        self.state_dict = {}
    
    def forward(self, x):
        pass
    
    def backward(self, dz):
        pass
        
    def zero_grad(self):
        pass
    
    def __call__(self, x):
        return self.forward(x)
