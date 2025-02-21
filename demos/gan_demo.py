from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

from dreamboat.utils.data_loader import DataLoader
from dreamboat.networks.sequential import Sequential
from dreamboat.networks.linear import Linear
from dreamboat.networks.relu import ReLU
from dreamboat.networks.leaky_relu import LeakyReLU
from dreamboat.networks.tanh import Tanh
from dreamboat.networks.sigmoid import Sigmoid
from dreamboat.losses.bce import BCELoss
from dreamboat.optimizers.adam import Adam


# load mnist dataset
(x_train_origin,t_train_origin),(x_test_origin,t_test_origin) = mnist.load_data()

# batch data loader
X_train,X_test = x_train_origin/255.,x_test_origin/255.
batch_size = 100

train_loader = DataLoader(X_train,t_train_origin,batch_size)
test_loader = DataLoader(X_test,t_test_origin,batch_size)
def normalize(x):
    return (x-x.min())/(x.max()-x.min())

# Hyper-parameters
z_dim = 64
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 100

# Discriminator
D = Sequential(
    Linear(image_size, hidden_size),
    LeakyReLU(0.2),
    Linear(hidden_size, hidden_size),
    LeakyReLU(0.2),
    Linear(hidden_size, 1),
    Sigmoid()
)

# Generator 
G = Sequential(
    Linear(z_dim, hidden_size),
    ReLU(),
    Linear(hidden_size, hidden_size),
    ReLU(),
    Linear(hidden_size, image_size),
    Tanh()
)

lr = 2e-4
d_optimizer = Adam(lr)
g_optimizer = Adam(lr)
D.apply_optim(d_optimizer)
G.apply_optim(g_optimizer)
loss_func = BCELoss()

# Start training
for epoch in range(num_epochs):
    for i in range(len(train_loader)):
        images,_ = train_loader.get_batch()
        images = images.reshape(images.shape[0],-1)
        real_labels = np.ones((batch_size,1))*0.9
        fake_labels = np.zeros((batch_size,1))
        
        D.zero_grad()
        # Train D with real images
        p = D(images)
        d_loss,dp = loss_func(p,real_labels)
        D.backward(dp)
        # Train D with fake images
        z = np.random.randn(batch_size, z_dim)
        fake_images = G(z)
        p = D(fake_images)
        d_loss2,dp = loss_func(p,fake_labels)
        d_loss += d_loss2
        D.backward(dp)
        d_optimizer.step()
        
        G.zero_grad()
        # Trains G
        z = np.random.randn(batch_size, z_dim)
        fake_images = G(z)
        p = D(fake_images)
        g_loss,dp = loss_func(p,real_labels)
        dx = D.backward(dp)
        G.backward(dx)
        g_optimizer.step()
        
    # Save sampled images
    fake_images = fake_images.reshape(fake_images.shape[0], 28, 28)
    print('epoch: %d, discriminator loss: %.4f, generator loss: %.4f'%(
        epoch+1,d_loss.item(),g_loss.item()))
    fig,axes = plt.subplots(2,8,figsize=(16,4))
    for i in range(16):
        img = normalize(fake_images[i])
        axes[i//8][i%8].imshow(img,cmap=plt.cm.gray)
    plt.show()
