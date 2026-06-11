import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter1d

def line(x,A,B):
    return A*x+B

def exp_inv(x,A,B,C):
    return A*np.exp(-C*x) + B

A = pd.read_csv("Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW4/Q2/Data/M.csv").to_numpy()

beta = A[:,0]/100
M = A[:,1]

M = median_filter(M, size = 11)
M = gaussian_filter1d(M, sigma= 2)

args, _ = curve_fit(exp_inv, beta, M, p0=[43,-38,2])

plt.figure('M')
plt.title(r'Magnetization M in funtion of $\beta$')
plt.xlabel(r'$\beta$')
plt.ylabel('M')
plt.plot(beta,M,'.', label = 'simulated M, noise filtered')
plt.plot(beta, exp_inv(beta, args[0], args[1], args[2]), label = f'fitted function: 'r'M($\beta$) ='f' {"%.2f"%args[0]}exp(-{"%.2f"%args[2]}'r'$\beta$'f') {"%.2f"%args[1]}')
plt.legend()
plt.show()

# A = pd.read_csv("Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW4/Q2/Data/E_avg_T.csv").to_numpy()

# dT = A[:,0]/100
# E_avg_T = A[:,1]

# C = np.gradient(E_avg_T,dT)
# C = median_filter(C, size = 11)
# C = gaussian_filter1d(C, sigma= 2)
# C = median_filter(C, size = 11)
# C = gaussian_filter1d(C, sigma= 2)
# C = median_filter(C, size = 11)
# C = gaussian_filter1d(C, sigma= 2)
# C = median_filter(C, size = 11)
# C = gaussian_filter1d(C, sigma= 2)
# C = median_filter(C, size = 11)
# C = gaussian_filter1d(C, sigma= 2)

# args, _ = curve_fit(line, dT, C)

# plt.figure('C')
# plt.title(r'Heat capacity C(T) in function of $\beta$')
# plt.xlabel(r'$\beta$')
# plt.ylabel('C(T)')
# plt.plot(dT,C,'.', label = 'simulated C(T), noise filtered')
# plt.plot(dT, args[0]*dT + args[1], label = f'linear fit: C(T) = {"%.2f"%args[0]}'r'$\beta$' f'{"%.2f"%args[1]}')
# plt.legend()
# plt.show()