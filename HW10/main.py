import numpy as np
import matplotlib.pyplot as plt 

N = 100

X_i = np.random.rand(N)

X_i_sqrd = X_i**2

X_i_sqrd_mean = X_i_sqrd.mean()

X_i_mean =  X_i.mean()

sigma_x_sqrd_1 = X_i_sqrd_mean - X_i_mean**2

indx = [0,1,2,3,4,5,6,7,8,9,10]

list_x = np.zeros(len(indx))

for i in range(len(indx)):
    list_x[i] = (X_i**i).mean()

# print((X_i_mean-0.5)/0.5, (X_i_sqrd_mean-1/3)/(1/3), (sigma_x_sqrd_1-1/12)/(1/12))

# plt.figure('mean_X_i_powers')
# plt.title(r'Graph of $<X_i^n>$ for different values of "n"')
# plt.xlabel('n')
# plt.ylabel(r'$<X_i^n>$')
# plt.plot(indx,list_x)
# plt.show()

N = np.array([10,20,40])
m = int(1e5)

sigma_10 = np.zeros(m)
sigma_20 = np.zeros(m)
sigma_40 = np.zeros(m)

sigma = np.array([sigma_10,sigma_20,sigma_40])


for j in range(len(N)):
    for i in range(m):
        X_i = np.random.rand(N[j])
        sigma_x_sqrd = ((X_i-X_i.mean())**2).mean()
        sigma[j,i] = sigma_x_sqrd

sigma_mean_approx =  sigma.sum(axis=1)/m
print(sigma_mean_approx)