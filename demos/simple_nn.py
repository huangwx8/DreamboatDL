from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

from dreamboat.utils.data_loader import DataLoader
from dreamboat.networks.sequential import Sequential
from dreamboat.networks.linear import Linear
from dreamboat.networks.relu import ReLU
from dreamboat.losses.cross_entropy import CrossEntropyLossWithSoftMax
from dreamboat.optimizers.adam import Adam


# load mnist dataset
(x_train_origin,t_train_origin),(x_test_origin,t_test_origin) = mnist.load_data()

# batch data loader
X_train,X_test = x_train_origin/255.,x_test_origin/255.
batch_size = 100

train_loader = DataLoader(X_train,t_train_origin,batch_size)
test_loader = DataLoader(X_test,t_test_origin,batch_size)

num_epochs = 5
total_step = len(train_loader)

model = Sequential(
    Linear(28*28,256),
    ReLU(),
    Linear(256,128),
    ReLU(),
    Linear(128,10)
)
loss_func = CrossEntropyLossWithSoftMax(10)
optimizer = Adam(0.001)
model.apply_optim(optimizer)

loss_list = []
log_step = 100 

for epoch in range(num_epochs):
    running_loss = 0.
    for i in range(total_step):
        x,y = train_loader.get_batch()
        x = x.reshape(x.shape[0],-1)
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
            running_loss/=log_step
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, running_loss))
            loss_list.append(running_loss)
            running_loss = 0.

plt.plot(loss_list)
plt.title('Train loss(Cross Entropy)')
plt.show()
