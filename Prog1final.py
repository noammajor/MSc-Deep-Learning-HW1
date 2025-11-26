#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 17:41:33 2025

@author: noammajor
"""
#%% import
import numpy as np
import matplotlib.pyplot as plt
#%%  dataset
# Create dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([-1, 1, 1, -1]).reshape(-1, 1)


#%% max function
def MaxFunc(U,x,b):
    z = np.dot(x, U) + b 
    return np.maximum(0, z), z
#%% graidents function
def Graidents(y,y_Pred ,w ,h , x):
    #dirivitive by yPred
    DL_dyPred = -2*(y - y_Pred) / y.shape[0]
    #graident by w
    DL_dw = np.dot(h.T, DL_dyPred)
    #graident by b2
    DL_db2 = np.sum(DL_dyPred, axis=0, keepdims=True)
    #Dz
    DzHelper = np.dot(DL_dyPred, w.T)* (h > 0).astype(float)
    #graident by b1
    DL_db1 = np.sum(DzHelper, axis=0, keepdims=True)
    #graident by U
    DL_dU = np.dot(x.T,DzHelper)
    return DL_dw,DL_db2,DL_db1,DL_dU
#%% Test function
def Test(U,X,b1,b2,y, w):
    #forward pass
    h, z = MaxFunc(U, X, b1)
    y_Pred = np.dot(h, w) + b2
    
    #loss
    loss = np.mean((y_Pred - y)**2)
    preds = np.where(y_Pred >= 0, 1, -1)
    accuracy = np.mean(np.equal(y, preds))
    return loss, accuracy
    
#%% Model Training

learningRate = 0.1
epochs = 400
input_dim = 2
internal_dim =2

# Random Init seed
np.random.seed(None)

#Random inishilization
#bais is safe at zero to start
U = np.random.normal(loc=0.0, scale=0.5, size=(input_dim, internal_dim))
b1 = np.zeros((1, internal_dim))
w = np.random.normal(loc=0.0, scale=0.5, size=(internal_dim, 1))
b2= np.zeros((1,1))

#all trained params for random tracking
all_param_random_track = [
    (U, 0, 0, "U[0,0]"), (U, 0, 1, "U[0,1]"), 
    (U, 1, 0, "U[1,0]"), (U, 1, 1, "U[1,1]"),
    (b1, 0, 0, "b1[0,0]"), (b1, 0, 1, "b1[0,1]"),
    (w, 0, 0, "w[0,0]"), (w, 1, 0, "w[1,0]"),
    (b2, 0, 0, "b2")
]

# choose randomly 2 parameters to track (np only, so cant use random libary)
indices = np.random.choice(len(all_param_random_track), 2, replace=False)

#random params to track
randomTrack1 = all_param_random_track[indices[0]]
randomTrack2 = all_param_random_track[indices[1]]

train_losses = []
param_1_history = []
param_2_history = []
for epoch in range(epochs):
    
    #forward pass
    h, z = MaxFunc(U, X, b1)
    y_Pred = np.dot(h,w) + b2
    
    #loss
    loss = np.mean((y_Pred - y)**2)
    train_losses.append(loss)
    
    param_1_history.append(randomTrack1[0][randomTrack1[1], randomTrack1[2]])
    param_2_history.append(randomTrack2[0][randomTrack2[1], randomTrack2[2]])
    
    #graidents
    DL_dw,DL_db2,DL_db1,DL_dU = Graidents(y, y_Pred, w, h, X)
    
    #Update learnable paramters
    
    U -= learningRate*DL_dU
    w -= learningRate*DL_dw
    b1 -= learningRate*DL_db1
    b2 -= learningRate*DL_db2
    

# %% 4. Results + testing
test_loss, test_acc = Test(U, X, b1, b2, y, w)

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Q4 - part 3
ax1.plot(train_losses, color='red')
ax1.set_title(f"Loss (Test: {test_loss:.4f})")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("MSE Loss")

#Q4 - part 4
ax2.plot(param_1_history, label=randomTrack1[3], color='blue')
ax2.plot(param_2_history, label=randomTrack2[3], color='green')
ax2.set_title("Evolution of Random Parameters")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Parameter Value")
ax2.legend()

plt.tight_layout()
plt.show()