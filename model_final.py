#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ### Load data

# #### Data train

# In[95]:


def dataloader(path_dataset):
    images = []
    for filename in os.listdir(path_dataset):
        file_path = os.path.join(path_dataset, filename)
    
        with Image.open(file_path) as image:
            images.append(image.copy())
    number_of_images = len(images)
    return images, number_of_images


# In[97]:


#train
images_train_frac, num_frac_train = dataloader("/zhome/99/3/215784/Desktop/archive/Dataset/train/fractured")
images_train_notfrac, num_nofrac_train = dataloader("/zhome/99/3/215784/Desktop/archive/Dataset/train/not fractured")
num_images = num_frac_train + num_nofrac_train


# In[ ]:


#validation
images_val_frac, num_frac_val = dataloader("/zhome/99/3/215784/Desktop/archive/Dataset/val/fractured")
images_val_notfrac, num_nofrac_val = dataloader("/zhome/99/3/215784/Desktop/archive/Dataset/val/not fractured")
num_images_val = num_frac_val + num_nofrac_val


# ### Tranform images to tensors

# In[50]:


compressed_size = 224
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert image to grayscale if not already
    transforms.Resize((compressed_size, compressed_size)),  # Resize image
    transforms.ToTensor(),  # Convert to tensor and scale [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize for grayscale
])


# In[51]:


transformed_images = []
for image in images_train_frac:
    transformed_images.append(transform(image))
for image in images_train_notfrac:
    transformed_images.append(transform(image))


# In[52]:


transformed_val_images = []
for image in images_val_frac:
    transformed_val_images.append(transform(image))
for image in images_val_notfrac:
    transformed_val_images.append(transform(image))

# ### Create batches/shuffle

# $\rightarrow$ Genererer en liste med 273 elementer (hver svarende til en batch). Hver liste indeholde 32 elementer (med billedindexer)

# In[111]:


np.random.seed(1)
images = torch.stack(transformed_images,dim = 0)

def create_batches(num_images, batch_size):
    indices = np.arange(num_images)
    np.random.shuffle(indices)  # Shuffle the indices
    return [indices[i:i + batch_size] for i in range(0, num_images, batch_size)]

batch_size = 16
batchindexes = create_batches(num_images, batch_size)


# In[112]:


images_val = torch.stack(transformed_val_images,dim = 0)

batch_size_val = 30
batchindexes_val = create_batches(num_images_val, batch_size_val)


# $\rightarrow $ Laver liste med 1'ere og 0'ere som svarer til billedernes originale rækkefølge

# In[113]:


fracture_yes_no = torch.cat((torch.ones(num_frac_train),torch.zeros(num_nofrac_train)), dim = 0).unsqueeze(0)
fracture_yes_no_val = torch.cat((torch.ones(num_frac_val),torch.zeros(num_nofrac_val)), dim = 0).unsqueeze(0)


# ### Formater batches og gør dem klar til modellen

# $\rightarrow $ Lægger batches i en 273 elementer lang liste. I hvert listeelement er der en tensor med 32 elementer, som hver er et billede.

# $\rightarrow$ Lægger 1'ere og 0'ere på et format der svarer til billederne

# In[114]:


batches = []
fracture_yes_no_batch = []

for minibatch in batchindexes:
    batches.append(torch.stack([images[index] for index in minibatch], dim = 0))
    fracture_yes_no_batch.append(torch.tensor([fracture_yes_no[0][num] for num in minibatch]).unsqueeze(1))


# In[115]:


batches_val = []
fracture_yes_no_batch_val = []

for minibatch in batchindexes_val:
    batches_val.append(torch.stack([images_val[index] for index in minibatch], dim = 0))
    fracture_yes_no_batch_val.append(torch.tensor([fracture_yes_no_val[0][num] for num in minibatch]).unsqueeze(1))


# ### Træn model

# #### Setup

# In[208]:


# Device to use for computations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use the nn package to define our model and loss function.

model = torch.nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # 32 kernels. dim in = 1* pic * pic, dimpot
    nn.BatchNorm2d(32),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),

    nn.Dropout(0.825),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), #64 kernels. dim in = pic/2 * pic/2 * 32
    nn.BatchNorm2d(64),
    nn.MaxPool2d(kernel_size=2),
    nn.ReLU(),

    nn.Dropout(0.825),

    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # 128 kernels. dim in = 1* pic * pic, dimpot
    nn.BatchNorm2d(128),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Dropout(0.825),
    

    nn.Flatten(), # dim = pic/4 * pic/4 * 64

    nn.Linear(in_features=100352, out_features=1),    # 1 fordi vi ønsker et tal mellem 0 og 1
    nn.Sigmoid(),
)
model.to(device)
loss_fn = torch.nn.BCELoss()


learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# #### Træning


# In[249]:


def reset_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

model.apply(reset_weights);


# In[250]:


# Allocate space for loss
loss_train = []
loss_val = []
train_accuracy = []
val_accuracy = []


# In[251]:


# Number of iterations
epochs = 25

# In[ ]:
# uncomment for edge masking

'''def edge_masking(img_tensor, mask_size=20):
    """
    Zero out the outer `mask_size` pixels of the image on each side.
    img_tensor should be of shape (C, H, W).
    """
    c, h, w = img_tensor.shape
    # Mask top and bottom rows
    img_tensor[:, :mask_size, :] = 0
    img_tensor[:, -mask_size:, :] = 0
    # Mask left and right columns
    img_tensor[:, :, :mask_size] = 0
    img_tensor[:, :, -mask_size:] = 0
    return img_tensor'''
# In[252]:


for t in range(epochs):
    total_correct = 0
    total_samples = 0
    loss_total = 0
    model.train()
    for batch, fracture_batch in tqdm(zip(batches, fracture_yes_no_batch), desc=f"Epoch {t+1} Batches", leave=False):

        # Move data to device
        batch = batch.to(device)
        fracture_batch = fracture_batch.to(device)

        # We need to apply edge_masking() to each image in the batch.
        '''for i in range(batch.size(0)):
            batch[i] = edge_masking(batch[i], mask_size=20)'''

        # Forward pass: compute predicted y by passing x to the model.
        fracture_pred = model(batch)
        
        # Compute and save loss.
        loss = loss_fn(fracture_pred, fracture_batch)
        loss_total += loss.item()

        correct = ((fracture_pred - fracture_batch).abs() <= 0.5).sum().item()  # Count correct predictions
        total_correct += correct
        total_samples += len(fracture_batch)  # Total number of samples in this batch
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step() 
    train_accuracy.append(total_correct/total_samples)
    loss_train.append(loss_total)

# FOR VALIDATION
    total_correct_val = 0
    total_samples_val = 0
    loss_total_val = 0
    model.eval()
    with torch.no_grad():
        for batch, fracture_batch_val in tqdm(zip(batches_val, fracture_yes_no_batch_val), desc=f"Epoch {t+1} Batches", leave=False):
            # Move data to device
            batch = batch.to(device)
            fracture_batch_val = fracture_batch_val.to(device)

            # Forward pass: compute predicted y by passing x to the model.
            fracture_pred = model(batch)
            # compute loss
            loss = loss_fn(fracture_pred, fracture_batch_val)
            loss_total_val += loss.item()

            # Compute and save loss.
            correct_val = ((fracture_pred - fracture_batch_val).abs() <= 0.5).sum().item()  # Count correct predictions
            total_correct_val += correct_val
            total_samples_val += len(fracture_batch_val)  # Total number of samples in this batch
        loss_val.append(loss_total_val)
        val_accuracy.append(total_correct_val/total_samples_val)


# In[257]:


torch.save(model.state_dict(), "model_weights_final_1.pth")

# In[258]:

loss_train = np.array(loss_train)
loss_val = np.array(loss_val)
# In[258]:
# save loss as plot


# Set x-axis range to start from 1
epochs = range(1, len(train_accuracy) + 1)

plt.plot(epochs,loss_train/num_images, marker='o')
plt.plot(epochs,loss_val/num_images_val, marker='o') 
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
'''for i, value in enumerate(loss_train):
    plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')
for i, value in enumerate(loss_val):
    plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')'''
#plt.show()
plt.savefig("loss_model_final_1.png")

plt.close()
# In[259]:

plt.ylim(0, 1)

plt.plot(epochs,train_accuracy, marker='o') 
plt.plot(epochs,val_accuracy, marker='o') 
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
'''for i, value in enumerate(val_accuracy):
    plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')
for i, value in enumerate(train_accuracy):
    plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')'''

# Annotate only the last accuracy value in the plot
plt.text(epochs[-1], val_accuracy[-1], f'{val_accuracy[-1]:.2f}', ha='center', va='bottom')

#plt.show()
plt.savefig("accuracy_model_final_1.png")