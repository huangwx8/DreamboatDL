from io import open
import glob
import string
import numpy as np
import random
import matplotlib.pyplot as plt

from dreamboat.networks.sequential import Sequential
from dreamboat.networks.rnn import RNN
from dreamboat.networks.linear import Linear
from dreamboat.losses.cross_entropy import CrossEntropyLossWithSoftMax
from dreamboat.optimizers.sgd import SGD


def findFiles(path):
    return glob.glob(path)

# keep lower case only
all_letters = string.ascii_lowercase
n_letters = len(all_letters)


# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    lines = [''.join(list(filter(str.isalpha, line))).lower() for line in lines]
    return lines

for filename in findFiles('names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    # 18 categories
    all_categories.append(category)
    lines = readLines(filename)
    # category_lines { category -> line of names }
    category_lines[category] = lines

n_categories = len(all_categories)

print("all_letters: "+str(len(all_letters)))
print("n_categories: "+str(n_categories))
print("category_lines:"+str(category_lines['Italian'][:5]))

# One hot encoding
# Find letter index from all_letters, e.g. "a" = 0
# Return position of letter
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToMatrix(line, seq_len):
    oh_mat = np.zeros((seq_len, n_letters))
    indices = [all_letters.find(letter) for letter in line]
    indices += [0]*(seq_len-len(indices))
    oh_mat[range(seq_len),indices] = 1
    return oh_mat

print(lineToMatrix('Jones',5))

# Sample randomly
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(n_samples):
    x_samples = []
    y_samples = []
    
    for _ in range(n_samples):
        category = randomChoice(all_categories)
        line = randomChoice(category_lines[category])
        x_samples.append(line)
        y_samples.append(all_categories.index(category))
    
    seq_len = max(len(l) for l in x_samples)
    
    line_mat = [lineToMatrix(l,seq_len) for l in x_samples]
    line_mat = np.stack(line_mat)
    return line_mat,np.array(y_samples),line,category

model = Sequential(
    RNN(n_letters,180,requires_clip=True),
    Linear(180,n_categories),
)
loss_func = CrossEntropyLossWithSoftMax(n_categories)
optimizer = SGD(0.001)
model.apply_optim(optimizer)
loss_list = []
running_loss = 0.
num_iters = 100000
log_step = 1000

for i in range(num_iters):
    x,y,name,category = randomTrainingExample(1)
    # Forward pass
    logits = model(x)
    # calculate loss
    loss,dlogits = loss_func(logits,y)
    # Backward
    model.zero_grad()
    model.backward(dlogits)
    # optimize
    optimizer.step()
    
    running_loss += loss.item()

    if (i+1) % log_step == 0:
        running_loss /= log_step
        print ('Step [{}/{}], Loss: {:.4f}'.format(i+1, num_iters, running_loss))
        loss_list.append(running_loss)
        guess = all_categories[np.argmax(logits,axis = 1)[0]]
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%s / %s %s'%(name,guess,correct))
        running_loss = 0.

plt.plot(loss_list)
plt.title('Train loss(Cross Entropy)')
plt.show()
