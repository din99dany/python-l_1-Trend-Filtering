import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la


### Fie problema de programare patratica:
##  min_v  (1/2)v^TQx + q^Tv
##    s.l. -lambda <= v <= lambda

## Codul rezolva problema de mai sus prin
## Metoda Gradient Proiectat


dataset = np.loadtxt(open('snp500.txt', 'rb'), delimiter=",", skiprows=1)  # y
n = dataset.size

eps = float(input("Introduceti o eroare pentru gradient:\n"))
lamda = float(input("Introduceti o valoare pentru eroare:\n"))

D = np.zeros((n - 2, n))

for i in range(n - 2):
    D[i][i] = 1
    D[i][i + 1] = -2
    D[i][i + 2] = 1

## Date: Q, q
Q = D @ D.T
q = -D @ dataset
a = np.array(np.random.rand(2, 1))
f = lambda x: 0.5 * x.T @ Q @ x + q.T @ x


def Proj(v):
    res = []
    for i in v:
        maxim = max(-lamda, min(lamda, i))
        res.append(maxim)
    return np.array(res)


### Constanta Lipschitz a gradientului
Lips = np.max(la.eigvals(Q))
alpha = 1/Lips

x_old = np.ones(n - 2)
x = Proj(x_old - alpha * (Q @ x_old + q))
criteriu_stop = la.norm(x - x_old)

k = 0
while (criteriu_stop > eps):
    print(k, criteriu_stop)
    # if(k >= 1000):
    #     break
    x_old = x

    ## Pas gradient
    grad = Q @ x + q
    y = x - alpha * grad

    ## Pas proiectie
    x = Proj(y)

    criteriu_stop = la.norm(x - x_old)
    k = k + 1

solMGP = dataset - D.T @ x.T


plt.plot(np.linspace(1, len(dataset), len(dataset)), dataset, 'k:', linewidth=1.0, label="dataset")
plt.plot(np.linspace(1, len(solMGP), len(solMGP)), solMGP,'b-', linewidth=2.0, label="l1")
plt.legend()
plt.show()

print(k)

def customY():
    res = []
    x = 1
    y = 10

    for i in range(1000):
        y = y + (np.random.rand(1)[0] - 0.5) * (np.random.rand(1)[0] * 150)
        res.append(y)

    return np.array(res)