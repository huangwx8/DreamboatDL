from .module import Module
from ..utils.image_processing import col2im, im2col
import numpy as np


class MaxPool2d(Module):
    def __init__(self, kernel_size = 2, stride = 2, padding=0):
        super().__init__()
        self._pool_h = kernel_size
        self._pool_w = kernel_size
        self._stride = stride
        self._pad = padding
        
        self._latest_input = None
        self._arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self._pool_h + 2*self._pad)//self._stride+1
        out_w = (W - self._pool_w + 2*self._pad)//self._stride+1
        # im2col
        col_x = im2col(x, self._pool_h, self._pool_w, self._stride, self._pad) 
        # col_x.shape = (N*out_h*out_w,C*ker*ker)
        col_x = col_x.reshape(-1, self._pool_h*self._pool_w) 
        # col_x.shape = (N*out_h*out_w*C,ker*ker)
        # calculate argmax on every row
        arg_max = np.argmax(col_x, axis=1)
        out = col_x[range(col_x.shape[0]),arg_max]
        # col_x.shape = (N*out_h*out_w*C,)
        # reshape to normal image
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) 
        # size = (N,C,out_h,out_w)

        self._latest_input = x
        self._arg_max = arg_max

        return out

    def backward(self, dz):
        N,C,out_h,out_w = dz.shape
        dz = dz.transpose(0, 2, 3, 1) 
        # size = (N,out_h,out_w,C)
        pool_size = self._pool_h * self._pool_w
        n = np.prod(dz.shape)
        dmax = np.zeros((n, pool_size))
        # keep argmax only
        dmax[np.arange(n), self._arg_max.flatten()] = dz.flatten()
        # size = (N*out_h*out_w*C,ker*ker)
        dcol = dmax.reshape(N * out_h * out_w, -1)
        # size = (N*out_h*out_w,C*ker*ker)
        # col2im
        dx = col2im(dcol, self._latest_input.shape, self._pool_h,
                    self._pool_w, self._stride, self._pad)
        # size = (N,C,H,W)
        return dx
        