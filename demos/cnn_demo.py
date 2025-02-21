from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

from dreamboat.utils.data_loader import DataLoader
from dreamboat.networks.sequential import Sequential
from dreamboat.networks.conv2d import Conv2d
from dreamboat.networks.maxpool2d import MaxPool2d
from dreamboat.networks.flatten import Flatten
from dreamboat.networks.linear import Linear
from dreamboat.networks.relu import ReLU
from dreamboat.losses.cross_entropy import CrossEntropyLossWithSoftMax
from dreamboat.optimizers.adam import Adam

# load cifar-10 dataset
(x_train_origin,t_train_origin),(x_test_origin,t_test_origin) = cifar10.load_data()

x_img_train_normalize = x_train_origin.astype('float32') / 255.0
x_img_test_normalize = x_test_origin.astype('float32') / 255.0
y_img_train = t_train_origin.flatten()
y_img_test = t_test_origin.flatten()

batch_size = 100
train_loader = DataLoader(x_img_train_normalize,y_img_train,batch_size)
test_loader = DataLoader(x_img_test_normalize,y_img_test,batch_size)

num_epochs = 10
total_step = len(train_loader)

# 3-layers convolutional network
# [conv->relu]->pool x 3 -> fc -> relu -> fc -> softmax
model = Sequential(
    Conv2d(in_channels=3, out_channels=32, kernel_size = 3,
           stride=1,padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2,stride=2),
    Conv2d(in_channels=32, out_channels=64, kernel_size = 3,
           stride=1,padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2,stride=2),
    Conv2d(in_channels=64, out_channels=128, kernel_size = 3,
           stride=1,padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2,stride=2),
    Flatten(),
    Linear(128*4*4,128),
    ReLU(),
    Linear(128,10),
)
loss_func = CrossEntropyLossWithSoftMax(10)
optimizer = Adam(0.001)
model.apply_optim(optimizer)

loss_list = []
log_step = 50

for epoch in range(num_epochs):
    running_loss = 0.
    for i in range(total_step):
        x,y = train_loader.get_batch()
        x = x.transpose(0,3,1,2)
        # Forward pass
        logits = model(x)
        # calculate loss
        loss,dlogits = loss_func(logits,y)
        gits = loss_func(logits,y)
        # Backward
        model.zero_grad()
        model.backward(dlogits)
        # optimize
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i+1) % log_step == 0:
            running_loss/=log_step
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, running_loss))
            loss_list.append(running_loss)
            running_loss = 0.
            
    correct = 0
    total = 0

    for i in range(len(test_loader)):
        x,y = test_loader.get_batch()
        x = x.transpose(0,3,1,2)
        logits = model(x)
        predicted = np.argmax(logits, axis = 1)
        total += y.shape[0]
        correct += (predicted == y).sum()

    print('Accuracy of the network on the 10000 test images: %.2f %%'%(100 * correct / total))

plt.plot(loss_list)
plt.title('Train loss(Cross Entropy)')
plt.show()
