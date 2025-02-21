from .module import Module
import numpy as np


class RNN(Module):
    '''
    Single hidden layer RNN, using two linear layers and a tanh activation function
    h_{t} = tanh(mm(x_{t},U)+mm(h_{t-1},W)+b)
    Provides a single output and backpropagation for the final time step output h_t
    '''
    def __init__(self, input_size, hidden_size, requires_clip = False):
        super().__init__()
        
        self._input_size = input_size
        self._hidden_size = hidden_size
        self.requires_clip = requires_clip
        
        self.weight_ih = np.random.randn(input_size,hidden_size)/(np.sqrt(input_size)/2)
        self.weight_hh = np.random.randn(hidden_size,hidden_size)/(np.sqrt(hidden_size)/2)
        self.bias = np.zeros(hidden_size)
        
        # Intermediate parameters
        self._x_t = None # input x at time t
        self._h_t = None # hidden h at time t
        self._h_t_no_tanh = None # x.dot(weight_ih)+h.dot(weight_hh)+bias at time t
        
        # gradients
        self._dw_ih = np.zeros_like(self.weight_ih)
        self._dw_hh = np.zeros_like(self.weight_hh)
        self._db = np.zeros_like(self.bias)
        
        self.state_dict['weight_ih'] = {'value':self.weight_ih,'grad':self._dw_ih}
        self.state_dict['weight_hh'] = {'value':self.weight_hh,'grad':self._dw_hh}
        self.state_dict['bias'] = {'value':self.bias,'grad':self._db}
        
    def forward(self, x, h_0 = None):
        '''
        x: np array, shape = (batch_size, seq_len, input_size)
        h_0: hidden unit, None for all zero
        out: h_t, shape = (batch_size, hidden_size)
        '''
        n,seq_len,m = x.shape
        assert m==self._input_size
        if h_0 is None:
            h_0 = np.zeros((n,self._hidden_size))
            
        self._x_t = x.copy()
        self._h_t = np.empty((n,seq_len+1,self._hidden_size))
        self._h_t_no_tanh = np.empty((n,seq_len,self._hidden_size))
        
        h_t = h_0
        for t in range(seq_len):
            x_t = x[:,t,:]
            self._h_t[:,t,:] = h_t
            self._h_t_no_tanh[:,t,:] = x_t.dot(self.weight_ih)+h_t.dot(self.weight_hh)+self.bias
            h_t = np.tanh(self._h_t_no_tanh[:,t,:])
        
        self._h_t[:,-1,:] = h_t
        self._seq_len = seq_len
        
        return h_t
    
    def backward(self, dh):
        '''
        dh: np array, shape = (batch_size, hidden_size)
        gradient of final hidden
        calculate bptt
        '''
        h_t = self._h_t[:,-1,:]
        for t in range(self._seq_len-1,-1,-1):
            do = dh*(1-h_t**2)
            
            h_t = self._h_t[:,t,:]
            x_t = self._x_t[:,t,:]
            
            dw_ih_t = x_t.T.dot(do)
            dw_hh_t = h_t.T.dot(do)
            db_t = np.sum(do,axis = 0)
            dh = do.dot(self.weight_hh.T)
            
            self._dw_ih += dw_ih_t
            self._dw_hh += dw_hh_t
            self._db += db_t
        
        if self.requires_clip:
            self.clip()
        
        return 'success'
    
    def clip(self, threshold = 2.):
        clipped_dw_ih = np.clip(self._dw_ih,-threshold,threshold)
        clipped_dw_hh = np.clip(self._dw_hh,-threshold,threshold)
        clipped_db = np.clip(self._db,-threshold,threshold)
        
        self.zero_grad()
        self._dw_ih += clipped_dw_ih
        self._dw_hh += clipped_dw_hh
        self._db += clipped_db
        
    def zero_grad(self):
        self._dw_ih -= self._dw_ih
        self._dw_hh -= self._dw_hh
        self._db -= self._db
