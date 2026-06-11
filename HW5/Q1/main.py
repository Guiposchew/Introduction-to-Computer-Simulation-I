import numpy as np
import matplotlib.pyplot as plt

def get_Xi(i, X0 = 1,a = 5,c = 0,m = 2048):
    X = np.zeros(i+1)
    X[0] = X0
    for j in range(i):
        X[j+1] = (a*X[j]+c)%m

    X_max = np.max(X)

    return X/X_max

def get_sigma(X,N):
    X_sm = np.sum(X**2)/(N)
    X_ms = (np.sum(X)/(N))**2

    return X_sm - X_ms

def get_running_average(X,N_max):
    X_r_avg = np.zeros(N_max)

    for i in range(1,N_max):
        X_r_avg[i] = np.sum(X[:i])/i

    return X_r_avg, X_r_avg[-1]


X = get_Xi(10**5,a = 273673163155,c=11, m=2**48)

N1 = 10**3
N2 = 10**4
N3 = 10**5

sigma1 = get_sigma(X[:N1],N1)
sigma2 = get_sigma(X[:N2],N2)
sigma3 = get_sigma(X[:N3],N3)

Xavg1 = get_running_average(X,N1)
Xavg2 = get_running_average(X,N2)
Xavg3 = get_running_average(X,N3)

sigma = [sigma1, sigma2, sigma3]
Xavg = [Xavg1, Xavg2, Xavg3]

t1 = np.arange(N1)
t2 = np.arange(N2)
t3 = np.arange(N3)

t = [t1,t2,t3]

plt.figure(f'Running_average for N = {N1}')
plt.title(f'Running average of {N1} Pseudo Random numbers')
plt.xlabel('Iteration')
plt.ylabel('Average')
plt.plot(t[0],Xavg1[0],'-',label = 'Running average')#, linewidth = 0.5)
plt.legend()
plt.figure(f'Running_average for N = {N2}')
plt.title(f'Running average of {N2} Pseudo Random numbers')
plt.xlabel('Iteration')
plt.ylabel('Average')
plt.plot(t[1],Xavg2[0],'r-',label = 'Running average')#, linewidth = 0.5)
plt.legend()
plt.figure(f'Running_average for N = {N3}')
plt.title(f'Running average of {N3} Pseudo Random numbers')
plt.xlabel('Iteration')
plt.ylabel('Average')
plt.plot(t[2],Xavg3[0],'-',color='orange',label = 'Running average')#, linewidth = 0.5)
plt.legend()
#plt.show()

Y = np.zeros(len(X))
t2 = np.arange(len(X))

for i in range(-1,len(X)-1):
    Y[i] = X[i+1]

t2 = [t2[::2]+1][0]
t2 = t2[:(len(t2)-1)]


plt.figure('X_Y')
plt.title("Plot of $X_i$ "r'$\times$'" $X_{i+1}$, for $i \in [1,3,5...]$")
plt.xlabel('$X_i$')
plt.ylabel('$X_{i+1}$')
plt.plot(X[t2],Y[t2], '.')
plt.show()