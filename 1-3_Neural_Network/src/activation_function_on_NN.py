import numpy as np
import matplotlib.pylab as plt

# The step function
def step_func(x):
    y = x > 0
    return y.astype(np.int) # astype = cast()
'''
x = np.array([-1.0, 1.0, 2.0])
print(x)
print("step func = ", step_func(x))
'''

# The sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
'''
x = np.array([-1.0, 1.0, 2.0])
print("sigmoid = ", sigmoid(x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
'''

# ReLU
def relu(x):
    return np.maximum(0, x)
'''
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.plot(x, y)
plt.ylim(-1.0, 5.3)
plt.show()
'''
