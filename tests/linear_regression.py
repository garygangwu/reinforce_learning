import numpy as np
import matplotlib.pyplot as plt

def polynomials_func(x):
    v = 1
    w = 1
    b = 1
    return v * x**2 + w * x + b

X = np.array([-2, -1.2, -0.2, 0, 0.3, 0.5, 1, 1.7], dtype=np.float32)

Y = polynomials_func(X)
print("X:", X)
print("Y:", Y)

# model prediction
def forward(v, w, d, x):
    return v * x * x + w * x + d

def loss(y, y_predicted):
    result = (y-y_predicted)**2
    result = result[result != 0]
    return result.mean()

def gradient_dl_dd(Y, y_predicted):
    return (y_predicted - Y).mean() * 2

def gradient_dl_dw(X, Y, y_predicted):
    return np.multiply(2*X, y_predicted - Y).mean()

def gradient_dl_dv(X, Y, y_predicted):
    x_x = np.multiply(X, X) 
    return np.multiply(2*x_x, y_predicted - Y).mean()

def draw_figure(X, Y, y_predicted):
    plt.figure()
    ax = plt.subplot()
    ax.scatter(X, Y)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.plot(X, y_predicted, 'green')
    plt.show()

learning_rate = 0.01
epochs = 1000

w = 0.5
d = 0.5
v = 0.5
for epoch in range(epochs):
    y_predicted = forward(v, w, d, X)
    l = loss(Y, y_predicted)
    dv = gradient_dl_dv(X, Y, y_predicted)
    dw = gradient_dl_dw(X, Y, y_predicted)
    db = gradient_dl_dd(Y, y_predicted)
    v = v - learning_rate * dv
    w = w - learning_rate * dw
    d = d - learning_rate * db
    
    
    if epoch % 1 == 0:
        print(f"epoch {epoch}: v={v:.3f}, w={w:.3f}, d={d:.3f}, loss={l:0.8f}")
        
    if epoch % 100 == 0:
        draw_figure(X, Y, y_predicted)