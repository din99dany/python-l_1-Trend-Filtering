import numpy as np
import cvxpy as cp
import scipy as scipy
import matplotlib.pyplot as plt

# Load time series data: S&P 500 price log.
dataset = np.loadtxt(open('data/snp500.txt', 'rb'), delimiter=",", skiprows=1)  # y
datasetLength = dataset.size

# Form second difference matrix.
e = np.ones((1, datasetLength))
D = scipy.sparse.spdiags(np.vstack((e, -2*e, e)), range(3), datasetLength - 2, datasetLength)

# Set regularization parameter.
lambdaValue = 100

# Solve l1 trend filtering problem.
x = cp.Variable(shape=datasetLength)
# l1 trend filtering
# aici
obj = cp.Minimize(0.5 * cp.sum_squares(dataset - x) + lambdaValue * cp.norm(D*x, 1))
# obj = cp.Minimize(0.5 * x.T @ D @ D.T @ x - dataset.T @ D.T @ x)
# H-P filtering
# obj = cp.Minimize(0.5 * cp.sum_squares(dataset - x) + lambdaValue * cp.norm(D*x, 2) ** 2)
prob = cp.Problem(obj, constraints=[-lambdaValue <= x, x <= lambdaValue])

# ECOS and SCS solvers fail to converge before
# the iteration limit. Use CVXOPT instead.
prob.solve(solver=cp.CVXOPT, verbose=True)
print('Solver status: {}'.format(prob.status))

# Check for error.
if prob.status != cp.OPTIMAL:
    raise Exception("Solver did not converge!")

print("optimal objective value: {}".format(obj.value))

# Plot estimated trend with original signal.
plt.figure(figsize=(6, 6))
plt.plot(np.arange(1, datasetLength + 1), dataset, 'k:', linewidth=1.0)
plt.plot(np.arange(1, datasetLength + 1), np.array(x.value), 'b-', linewidth=2.0)
plt.xlabel('date')
plt.ylabel('log price')
plt.show()