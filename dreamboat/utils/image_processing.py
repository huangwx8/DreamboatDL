import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    '''
    input images, shape = (N,C,H,W)
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    output col, shape = (N*out_h*out_w,C*filter_h*filter_w)
    '''
    # Padding
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1  # height
    out_w = (W + 2*pad - filter_w)//stride + 1  # width
    
    img = np.pad(input_data,((0,0),(0,0),(pad,pad),(pad,pad)),"constant") # padding
    # shape = (N,C,H+2*pad,W+2*pad)
    col = np.empty((filter_h,filter_w,N,C,out_h,out_w))
    for y in range(filter_h):
        for x in range(filter_w):
            col[y, x] = img[:,:,y:out_h*stride+y:stride,x:out_w*stride+x:stride]
    col = col.transpose(2, 4, 5, 3, 0, 1).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    '''
    inverse operation to im2col
    input col, shape = (N*out_h*out_w,C*filter_h*filter_w)
    output images, shape = (N,C,H,W)
    '''
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    # shape = (N*out_h*out_w, C*filter_h*filter_w)
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(4, 5, 0, 3, 1, 2)
    # now: shape = (filter_h, filter_w, N, C, out_h, out_w)
    
    img = np.zeros((N, C, H+2*pad, W+2*pad))
    for y in range(filter_h):
        for x in range(filter_w):
            img[:,:,y:out_h*stride+y:stride,x:out_w*stride+x:stride] += col[y, x]

    return img[:, :, pad:H+pad, pad:W+pad]
