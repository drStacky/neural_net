# n Layer neural network (input, hidden layer, output)
# Based on tutorial: https://iamtrask.github.io/2015/07/12/basic-python-network/
import numpy as np
import matplotlib.pyplot as plt     # For plotting results


# Activation function
def sigmoid(f, deriv=False):
    if deriv:
        return f*(1-f)
    else:
        return 1/(1+np.exp(-f))


# Training set inputs
# Perfect correlation b/w 1XOR2 and output
x = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
# Training set outputs
y = np.array([[0, 1, 1, 0]]).T
# Seed random numbers for predictability in testing code
np.random.seed(314)


# Total number of layers (first layer is 0)
n = int(input("How many hidden layers? "))

# Synapse, this matrix is the connection between input and out that "learns"
# Start with random values between -1 and 1 (mean 0)
syn = []
for i in range(0, n):
    syn.append(2 * np.random.random((3, 3)) - 1)
syn.append(2 * np.random.random((3, 1)) - 1)

layers = []
errors = [[], []]

for epoch in range(0, 10000):
    # Make guess of output
    # Feed through hidden layers
    layers = [x]
    for j in range(0, n+1):
        layers.append(sigmoid(np.dot(layers[j], syn[j])))

    # Calculate error in guess
    error = y - layers[n + 1]

    # Save epoch vs error for plotting later
    errors[0].append(epoch)
    errors[1].append(np.linalg.norm(error))

    # dJ/dtheta
    delta = [y]*(n+1)
    delta[n] = error * sigmoid(layers[n + 1], True)
    for j in range(n-1, -1, -1):
        delta[j] = np.dot(delta[j+1], syn[j+1].T) * sigmoid(layers[j+1], True)

    # Synapse learns
    for j in range(0, n+1):
        syn[j] += np.dot(layers[j].T, delta[j])

# Final guess
print(layers[n + 1])

# Plot the error (should be decreasing towards 0)
plt.figure()
plt.scatter(errors[0], errors[1])
plt.title('Neural Net Error Convergence')
plt.xlabel('Epoch')
plt.ylabel('Error magnitude')
plt.show()
