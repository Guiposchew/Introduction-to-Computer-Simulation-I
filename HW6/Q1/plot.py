import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as sp

def exp_inv(x,A,B,C):
    return A*np.exp(-C*x) + B

A = pd.read_csv("Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW6/Q1/E_f.csv").to_numpy()
B = pd.read_csv("Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW6/Q1/E_f[4 4].csv").to_numpy()
C = pd.read_csv("Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW6/Q1/E_f[8 8].csv").to_numpy()
D = pd.read_csv("Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW6/Q1/E_f[16 16].csv").to_numpy()


beta = A[:,0]/1000
E = np.array([A[:,1],B[:,1],C[:,1],D[:,1]])
sizes = np.array([[50,50],[4,4],[8,8],[16,16]])

for i in range(0,4):
    args, _ = sp.curve_fit(exp_inv, beta, E[i])

    E_predict = exp_inv(beta,args[0], args[1], args[2])

    plt.figure(f'E{i}')
    plt.title(f'Energy per spin of a {sizes[i,0]} by {sizes[i,0]} lattice:')
    plt.xlabel(r'$\beta$')
    plt.ylabel('Energy')
    plt.plot(beta,E[i]/(sizes[i,0]**2),'.', label = 'calculated value of E')
    plt.plot(beta,E_predict/(sizes[i,0]**2), label = f'fitted function: 'r'E($\beta$) ='f' {"%.2f"%(args[0]/(sizes[i,0]**2))}exp(-{"%.2f"%args[2]}'r'$\beta$'f') {"%.2f"%(args[1]/(sizes[i,0]**2))}')
    plt.legend(loc = 'upper right')

    C = (beta**2)*(np.mean(E[i]**2)-(np.mean(E[i]))**2)/(sizes[i,0]**2)

    print(np.std(E[i]-E_predict))

    plt.figure(f'C{i}')
    plt.title(f'Heat capacity of a {sizes[i,0]} by {sizes[i,0]} lattice:')
    plt.xlabel(r'$\beta$')
    plt.ylabel('Heat Capacity')
    plt.plot(beta, C, label = 'Heat capacity')
    plt.legend(loc = 'upper left')
    plt.show()