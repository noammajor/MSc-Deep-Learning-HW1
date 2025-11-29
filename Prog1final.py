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

#%% helpers function
# gradients function
def Graidents(y,y_Pred ,w ,h , x):
    #dirivitive by yPred
    grad_y = -2*(y - y_Pred)
    #graident w
    grad_w = np.dot(h.T, grad_y)
    #graident b2
    grad_b2 = np.sum(grad_y, axis=0, keepdims=True)
    #max grad
    gradMaxF = np.dot(grad_y, w.T)* np.where(h > 0, 1, 0)
    #graident b1
    grad_b1 = np.sum(gradMaxF, axis=0, keepdims=True)
    #graident U
    grad_U = np.dot(x.T,gradMaxF )
    return grad_w,grad_b2,grad_b1,grad_U

# max func (forward pass)
def MaxFP(U,x,b):
    z = np.dot(x, U) + b 
    return np.maximum(0, z)
#%% Test function
def Test(U,X,b1,b2,y, w):
    #forward pass
    h = MaxFP(U, X, b1)
    y_Pred = np.dot(h, w) + b2
    
    #loss calc
    loss = np.mean((y_Pred - y)**2)
    preds = np.where(y_Pred >= 0, 1, -1)
    accuracy = np.mean(np.equal(y, preds))
    return loss, accuracy
    
#%% Model 

learningRate = 0.1
epochs = 400
input_dim = 2
internal_dim =2

# Random Init seed
np.random.seed(42)

#Random inishilization
#bais is safe at zero to start
U = np.random.normal(loc=0.0, scale=0.5, size=(input_dim, internal_dim))
b1 = np.zeros((1, internal_dim))
w = np.random.normal(loc=0.0, scale=0.5, size=(internal_dim, 1))
b2= np.zeros((1,1))

#all trained params for random tracking
all_param_random_track = [[U, (0, 0), "U[0,0]"], [U, (0, 1), "U[0,1]"], 
    [U, (1, 0) , "U[1,0]"], [U, (1, 1), "U[1,1]"],
    [b1, (0, 0), "b1[0,0]"], [b1, (0, 1), "b1[0,1]"],
    [w, (0, 0), "w[0,0]"], [w, (1, 0), "w[1,0]"],
    [b2, (0, 0), "b2"]]

# choose randomly 2 parameters to track (np only, so cant use random libary)
indices = np.random.choice(len(all_param_random_track), 2, replace=False)

#random params to track
ranTrack1 = all_param_random_track[indices[0]]
ranTrack2 = all_param_random_track[indices[1]]

train_losses = []
param_1_history = []
param_2_history = []
for epoch in range(epochs):
    
    #forward pass
    h = MaxFP(U, X, b1)
    y_Pred = np.dot(h,w) + b2
    
    #loss
    train_losses.append(np.mean((y_Pred - y)**2))
    
    param_1_history.append(ranTrack1[0][ranTrack1[1]])
    param_2_history.append(ranTrack2[0][ranTrack2[1]])
    grad_w,grad_b2,grad_b1,grad_U = Graidents(y, y_Pred, w, h, X)
    
    #Update learnable paramters
    U -= learningRate * grad_U
    w -= learningRate * grad_w
    b1 -= learningRate * grad_b1
    b2 -= learningRate * grad_b2

# Final Test + Plots
test_loss, test_acc = Test(U, X, b1, b2, y, w)
# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Q4 - part 3
ax1.set_xlabel("Epochs")
ax1.set_ylabel("MSE Loss")
ax1.set_title(f"Loss (Test: {test_loss:.4f})")
ax1.plot(train_losses, color='red', label='Training Loss')
ax1.grid()
ax1.legend()

#Q4 - part 4
ax2.set_title("Evolution of Random Parameters")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Parameter Value")
ax2.plot(param_1_history, label=ranTrack1[2], color='blue',)
ax2.plot(param_2_history, label=ranTrack2[2], color='green')
ax2.grid()
ax2.legend()

plt.tight_layout()
plt.show()
# %%
