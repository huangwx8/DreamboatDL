import numpy as np


class Adam:
    '''
    Adam optimizer
    update equation:
    s <- beta1*d+(1-beta1)*g
    r <- beta2*(g**2)+(1-beta)*(g**2)
    d <- s/r
    w <- w - lr*d
    '''
    def __init__(self,LEARNING_RATE=0.01,BETA1 = 0.9,BETA2 = 0.999):
        self.modules = []
        self.direct = []
        self.t = 0
        self.lr = LEARNING_RATE
        self.beta1 = BETA1
        self.beta2 = BETA2
        
    def add_module(self, module):
        self.modules.append(module)
        self.direct.append(dict())
        for key in module.state_dict:
            param = module.state_dict[key]['value']
            self.direct[-1][key] = {'s':np.zeros_like(param),
                                    'r':np.zeros_like(param)}
    
    def step(self):
        self.t += 1
        for i in range(len(self.modules)):
            for key in self.modules[i].state_dict:
                g = self.modules[i].state_dict[key]['grad']
                self.direct[i][key]['s'] *= self.beta1
                self.direct[i][key]['r'] *= self.beta2
                self.direct[i][key]['s'] += (1-self.beta1)*g
                self.direct[i][key]['r'] += (1-self.beta2)*(g**2)
                
                s_hat = self.direct[i][key]['s']/(1-self.beta1**self.t)
                r_hat = self.direct[i][key]['r']/(1-self.beta2**self.t)
                
                d = s_hat/(np.sqrt(r_hat)+1e-8)
                self.modules[i].state_dict[key]['value'] -= self.lr*d
        return 'success'
