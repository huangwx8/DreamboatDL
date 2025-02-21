from .module import Module
from ..utils.image_processing import col2im, im2col
import numpy as np


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3,
                 stride=1, padding=0):
        '''
        Initialize the weights (4-dimensional convolution kernels), biases, stride, and padding.
        '''
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._pad = padding
        
        
        fan_in = in_channels*kernel_size*kernel_size
        
        a = np.sqrt(5).item()
        bound = np.sqrt(6/((1+a**2)*fan_in)).item()
        # trainable parameters
        self.weights = (np.random.rand(out_channels,in_channels,
                    kernel_size,kernel_size)-0.5)*2*bound
        
        self.bias = np.zeros(out_channels)
        
        self._dw = np.zeros_like(self.weights)
        self._db = np.zeros_like(self.bias)
        
        self.state_dict['weights'] = {'value':self.weights,'grad':self._dw}
        self.state_dict['bias'] = {'value':self.bias,'grad':self._db}
        
        # intermediate
        self._latest_input = None   
        self._col_x = None
        self._col_weights = None
        
    def forward(self, x):
        # data dimensions
        N, C, H, W = x.shape
        # convolution kernel dimensions
        FN, C, FH, FW = self.weights.shape
        # output dimensions
        out_h = 1 + (H + 2*self._pad - FH) // self._stride
        out_w = 1 + (W + 2*self._pad - FW) // self._stride
        # im2col
        self._col_x = im2col(x, FH, FW, self._stride, self._pad)
        # reshape to 2D array
        self._col_weights = self.weights.reshape(FN, -1).T
        # feedforward
        out = np.dot(self._col_x, self._col_weights) + self.bias
        # out.shape = (N*out_h*out_w,FN)
        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2) 
        # out.shape = (N,FN,out_h,out_w)

        self._latest_input = x
        
        return out

    def backward(self, dz):
        # convolution kernel dimensions
        FN, C, FH, FW = self.weights.shape
        # dz.shape = (N,FN,out_h,out_w)
        dz = dz.transpose(0,2,3,1).reshape(-1, FN)
        # size = (N*out_h*out_w,FN)

        g_of_b = np.sum(dz, axis=0)
        # size(FN,) 与bias的维度相同
        g_of_w = np.dot(self._col_x.T, dz)
        # size = mm(size(C*ker*ker,N*out_h*out_w), (N*out_h*out_w,FN))
        # = size(C*ker*ker, FN)
        g_of_w = g_of_w.transpose(1, 0).reshape(FN, C, FH, FW)
        # size(FN,C,ker,ker), 与weights的维度相同
        
        self._dw += g_of_w
        self._db += g_of_b

        g_of_col_x = np.dot(dz, self._col_weights.T)
        # size = mm(size(N*out_h*out_w,FN), (FN,C*ker*ker))
        # = size(N*out_h*out_w, C*ker*ker)
        # col2im
        dx = col2im(g_of_col_x, self._latest_input.shape, FH, FW, self._stride, self._pad)
        # size(N,C,H,W) = self.x.shape

        return dx
        
    def zero_grad(self):
        # zero inner parameters
        self._dw -= self._dw
        self._db -= self._db
        