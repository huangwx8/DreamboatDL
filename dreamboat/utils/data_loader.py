from math import ceil
import numpy as np


class DataLoader:
    def __init__(self, data, targets, batch_size, shuffle = False):
        indices = np.arange(data.__len__())
        if shuffle:
            np.random.shuffle(indices)
        self.data = data[indices]
        self.targets = targets[indices]
        self.batch_size = batch_size
        self.index = 0
        
    def get_batch(self):
        x = self.data[self.index:self.index+self.batch_size]
        y = self.targets[self.index:self.index+self.batch_size]
        self.index += self.batch_size
        if self.index>=self.data.__len__():
            self.index = 0
        return x,y
    
    def __len__(self):
        return ceil(self.data.__len__()/self.batch_size)
