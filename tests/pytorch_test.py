import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

def polynomials_func(x):
    v = -19.9
    w = 5.5
    b = 12.56
    return v * x * x + w * x + b

X = np.array([], dtype=np.float32)

for i in range(50):
    value = (i - 25) / 7.1 + 0.1
    X = np.append(X, np.float32(value))
Y = polynomials_func(X)
print("X:", X)
print("Y:", Y)

def draw_figure(X, Y, y_predicted):
    plt.figure()
    ax = plt.subplot()
    ax.scatter(X, Y, s=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.plot(X, y_predicted, 'green')
    plt.show()

x_train = X.reshape(-1, 1)
y_train = Y.reshape(-1, 1)
    

learningRate = 0.01 
epochs = 10000

#model = PolynomialRegression(inputDim, outputDim)
# Define the model
model = nn.Sequential(
    nn.Linear(1, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 1)
)
print(model)

loss_fn = torch.nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

# Converting inputs and labels to Variable
inputs = Variable(torch.from_numpy(x_train))
for epoch in range(epochs+1):
    labels = Variable(torch.from_numpy(y_train))
    
    # Clear gradient buffers 
    # because we don't want any gradient from previous epoch 
    # to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()
    
    # get output from the model, given the inputs
    outputs = model(inputs)
    
    # get loss for the predicted output
    loss = loss_fn(outputs, labels)
    
    # get gradients w.r.t to parameters
    loss.backward()
    
    # update parameters
    optimizer.step()

    if epoch % 5000 == 0:
        print(f"epoch {epoch}, loss {loss.item()}")
    

XX = np.array([], dtype=np.float32)
for i in range(200):
    value = (i - 80) / 12.0 + 0.1
    XX = np.append(XX, np.float32(value))
YY = polynomials_func(XX)

with torch.no_grad():
    inputs = Variable(torch.from_numpy(XX.reshape(-1, 1)))
    y_predicted = model(inputs).data.numpy()
y_predicted = y_predicted.reshape(-1)
draw_figure(XX, YY, y_predicted)

