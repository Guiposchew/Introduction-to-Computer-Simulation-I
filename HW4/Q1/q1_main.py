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

    return X_r_avg

X = get_Xi(10**5,c=1, m=10*2**15)


tau = np.mean(np.diff(np.nonzero(X==min(X)))[0])

print(X)


# t = np.arange(len(X))
# X_m_tau = np.sum(X[:tau])/tau
# sigma_X_tau = get_sigma(X[:tau],tau)

# print("%.3f" %X_m_tau,"%.3f" %sigma_X_tau)

# X_r_avg = get_running_average(X, 40000)

# t = np.linspace(30000,40000,10000)

# plt.figure('Running_average')
# plt.title('Running average of Pseudo Random numbers, $n \in [30000, 40000]$')
# plt.xlabel('Seed (n)')
# plt.ylabel('Average')
# plt.plot(t,X_r_avg[30000:40000],'-',label = 'Running average')#, linewidth = 0.5)
# plt.legend()
# plt.show()

# Y = np.zeros(len(X))

# for i in range(-1,len(X)-1):
#     Y[i] = X[i+1]

# t = [t[::2]+1][0]
# t = t[:(len(t)-1)]

# plt.figure('X_Y')
# plt.title("Plot of $X_i$ "r'$\times$'" $X_{i+1}$, for $i \in [1,3,5...]$")
# plt.xlabel('$X_i$')
# plt.ylabel('$X_{i+1}$')
# plt.plot(X[t],Y[t], '.')
# plt.show()