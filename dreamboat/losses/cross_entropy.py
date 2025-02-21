import numpy as np


class CrossEntropyLossWithSoftMax:
    '''
    softmax, logits to probabilities
    using cross entropy loss to calculate gradient
    '''
    def __init__ (self, out_features):
        super().__init__()
        self.out_features = out_features
        
    def one_hot_encode(self, labels, num_classes):
        '''
        labels: 1 dim array
        return: one hot matrix 
        '''
        assert labels.max()+1<=num_classes
        
        batch_size = len(labels)
        oh_mat = np.zeros((batch_size,num_classes))
        oh_mat[range(batch_size),labels] = 1.
        return oh_mat
    
    def get_prob(self, x):
        '''
        calculate probabilities
        '''
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x,axis=1,keepdims=True)
        return exp_x/sum_exp_x
        
    def __call__(self, x, labels):
        '''
        cross entropy loss
        loss(l) = -log(p_l)
        '''
        batch_size = len(labels)
        y_hat = self.get_prob(x)
        loss = -np.mean(np.log(y_hat[range(batch_size),labels]))
        dx = y_hat-self.one_hot_encode(labels,self.out_features)
        return loss,dx/batch_size
