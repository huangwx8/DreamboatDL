from .module import Module
import numpy as np


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = np.random.randn(num_embeddings, embedding_dim)
        self._dw = np.zeros_like(self.weight)
        self.state_dict['weight'] = {'value':self.weight,'grad':self._dw}
        
    def forward(self, x):
        '''
        x: np.array: dtype=int, shape=whatever
        return: np.array: dtype=float, shape=whatever+(embedding_dim,)
        '''
        self._x = x
        x_shape = x.shape
        x = x.flatten()
        out = self.weight[x]
        out = out.reshape(x_shape+(self.embedding_dim,))
        return out

    def backward(self, dz):
        '''
        dz: shape = whatever+(embedding_dim,)
        calculate gradient of weight
        hits: generally, indices is int type, so we cant get dx
        '''
        dz = dz.reshape(-1,self.embedding_dim)
        x = self._x.flatten()
        
        for i in range(len(x)):
            self._dw[x[i]] += dz[i]

        return None
        
    def zero_grad(self):
        self._dw -= self._dw
