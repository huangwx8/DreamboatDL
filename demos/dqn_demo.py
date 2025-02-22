import numpy as np
import matplotlib.pyplot as plt
import gym
import random
from collections import deque

from dreamboat.networks.sequential import Sequential
from dreamboat.networks.linear import Linear
from dreamboat.networks.tanh import Tanh
from dreamboat.optimizers.adam import Adam
from dreamboat.losses.mse import MSELoss

ENV_NAME = 'MountainCar-v0'
env = gym.make(ENV_NAME)

MAX_STEPS = 200

# Observations are continuous, (speed, position)
print(env.observation_space)

# Actions are discrete, (go left, go right, do nothing)
print(env.action_space)

action_num = 3

class DQNAgent():
    '''
    Network will evaluate Q values of actions for next step under current state, aka Q(s,a) = nn(s)[a]
    Network will be trained according to "Q(s0,a) = r + gamma*max_i(Q(s1,a_i))"
    In training progress, we use epsilon greedy policy to make decisions, and update environment
    '''
    def __init__(self, network, eps, gamma, lr):
        self.network = network
        self.eps = eps
        self.gamma = gamma
        self.optimizer = Adam(lr)
        self.loss_func = MSELoss()
        self.network.apply_optim(self.optimizer)

    def learn(self, batch):
        s0, a0, r1, s1, done = zip(*batch)
        s0,a0,r1,s1 = np.array(s0),np.array(a0),np.array(r1),np.array(s1)
        
        n = len(s0)
        
        y_true = r1 + self.gamma * np.max(self.network(s1), axis=1)
        y_true[done] = r1
        y_pred = self.network(s0)
        y_true_mat = y_pred.copy()
        y_true_mat[range(n),a0] = y_true
        self.network.zero_grad()
        loss,dy = self.loss_func(y_pred,y_true_mat)
        self.network.backward(dy)
        self.optimizer.step()

    def sample(self, state):
        '''
        Eplision greedy policy
        '''
        state = state[np.newaxis,:]
        action_value = self.network(state)
        
        if random.random()<self.eps:
            return random.randint(0, action_num-1)
        else:
            max_action = np.argmax(action_value,axis=1)
            return max_action.item()
        
def evaluate(agent, env, times):
    '''
    The evaluation function uses two metrics: 
    one is the furthest position the agent reaches in the game, and the other is the game time. 
    This environment will only stop once we successfully reach position = 0.5; 
    otherwise, we have to wait at least 200 time units. 
    Therefore, the time should be as small as possible, and the furthest position should be as large as possible.
    '''
    score = 0.
    avg_t = 0.
    eps = agent.eps
    agent.eps = 0
    
    for _ in range(times):
        s, _ = env.reset()
        done = False
        max_pos = -100
        t = 0
        
        while not done:
            t += 1
            max_pos = max(s[0],max_pos)
            a = agent.sample(s)
            s,r,done,_,_ = env.step(a)
            
            if t >= MAX_STEPS:
                done = True
            
            if done:
                score += max_pos
                avg_t += t
    
    env.reset()
    agent.eps = eps
    return score/times,avg_t/times

replays = deque(maxlen=10000)
gamma = 0.99
eps_high = 0.3
eps_low = 0.01
num_episodes = 500
LR = 0.001
batch_size = 10
log_step = 10

net = Sequential(
    Linear(2, 32),
    Tanh(),
    Linear(32,64),
    Tanh(),
    Linear(64,3)
)
agent = DQNAgent(net,1,gamma,LR)

score_list = []
time_cost_list = []

for episode in range(num_episodes):
    agent.eps = eps_high - (eps_high-eps_low)*(episode/num_episodes)
    
    s0, _ = env.reset()
    done = False
    max_pos = -100
    t = 0
    
    while not done:
        t += 1
        max_pos = max(s0[0],max_pos)
        a0 = agent.sample(s0)
        s1, r1, done, _, _ = env.step(a0)
        
        if t >= MAX_STEPS:
            done = True
        
        # Use max_pos as reward
        if done:
            if max_pos >= 0.3:
                # If max_pos is already large enough, add t as additional reward
                r1 = (max_pos+0.5) + (1-t/200)
            else:
                r1 = (max_pos+0.5)
        
        
        # Sample replay
        if not done:
            replay_rate = 0.1
        else:
            replay_rate = 1
        
        if random.random()<replay_rate:
            replays.append((s0.copy(),a0,r1,s1.copy(),done))
        
        s0 = s1
    
        if replays.__len__()>=batch_size:
            batch = random.sample(replays,k=batch_size)
            agent.learn(batch)
        
    if (episode+1)%log_step==0:
        score,t = evaluate(agent,env,10)
        score_list.append(score)
        time_cost_list.append(t)
        print("Episode: %d, Score: %.3f, average time cost: %.3f"%(episode+1,score,t))

plt.figure()
plt.plot(score_list)
plt.title('score')
plt.figure()
plt.plot(time_cost_list)
plt.title('average time cost')
plt.show()
