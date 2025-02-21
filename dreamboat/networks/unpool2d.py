from .module import Module
from ..utils.image_processing import col2im, im2col
import numpy as np


class UnPool2d(Module):
    def __init__(self, kernel_size = 2, stride = 2, padding=0):
        super().__init__()
        self._pool_h = kernel_size
        self._pool_w = kernel_size
        self._stride = stride
        self._pad = padding
        
        self._x = None
        self._arg_max = None

    def forward(self, x):
        self._x = x
        N, C, out_h, out_w = x.shape
        H = (out_h-1)*self._stride+self._pool_h-2*self._pad
        W = (out_w-1)*self._stride+self._pool_w-2*self._pad
        
        col_x = x.transpose(0,2,3,1).reshape(-1,1)
        col_x = np.tile(col_x,reps=(1,self._pool_h*self._pool_w))
        out = col2im(col_x, (N,C,H,W), self._pool_h,
                    self._pool_w, self._stride, self._pad)
        return out

    def backward(self, dz):
        N, C, out_h, out_w = self._x.shape
        H = (out_h-1)*self._stride+self._pool_h-2*self._pad
        W = (out_w-1)*self._stride+self._pool_w-2*self._pad
        dz = im2col(dz, self._pool_h, self._pool_w, self._stride, self._pad)
        dz = dz.reshape(N*out_h*out_w*C,-1)
        dz = np.sum(dz,axis = 1)
        dz = dz.reshape(N,out_h,out_w,C).transpose(0,3,1,2)
        return dz
