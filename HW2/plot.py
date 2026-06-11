import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

A = pd.read_csv("Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW2/Z5.csv", sep =';')
B = pd.read_csv("Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW2/Z10.csv", sep =';')
C = pd.read_csv("Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW2/Z20.csv", sep =';')

Z5 = A['Partition function'].to_numpy()[~np.isnan(A['Partition function'].to_numpy())]
beta5 = A['Beta'].to_numpy()[~np.isnan(A['Beta'].to_numpy())]
E_i5 = A['Energy of each Configuration'].to_numpy()[~np.isnan(A['Energy of each Configuration'].to_numpy())]
E_unique5 = A['Unique energy levels'].to_numpy()[~np.isnan(A['Unique energy levels'].to_numpy())]

Z10 = B['Partition function'].to_numpy()[~np.isnan(B['Partition function'].to_numpy())]
beta10 = B['Beta'].to_numpy()[~np.isnan(B['Beta'].to_numpy())]
E_i10 = B['Energy of each Configuration'].to_numpy()[~np.isnan(B['Energy of each Configuration'].to_numpy())]
E_unique10 = B['Unique energy levels'].to_numpy()[~np.isnan(B['Unique energy levels'].to_numpy())]

Z20 = C['Partition function'].to_numpy()[~np.isnan(C['Partition function'].to_numpy())]
beta20 = C['Beta'].to_numpy()[~np.isnan(C['Beta'].to_numpy())]
E_i20 = C['Energy of each Configuration'].to_numpy()[~np.isnan(C['Energy of each Configuration'].to_numpy())]
E_unique20 = C['Unique energy levels'].to_numpy()[~np.isnan(C['Unique energy levels'].to_numpy())]

Z = [Z5, Z10, Z20]
#T = [1/beta5, 1/beta10, 1/beta20]
indx = [5,10,20]

Z_t5 = (2**5)*(np.cosh(beta5)**4)
Z_t10 = (2**10)*(np.cosh(beta10)**9)
Z_t20 = (2**20)*(np.cosh(beta20)**19)

Z_t = [Z_t5, Z_t10, Z_t20]



for i in range(0,len(indx)):
    plt.figure(i)
    plt.title(r'$Z \times k_bT$, for ' f'N = {indx[i]}')
    plt.xlabel(r'$k_bT$')
    plt.ylabel('#')
#    plt.plot(T[i], Z[i], label = f' Calculated result for N = {indx[i]}')
    # plt.plot(T[i], Z_t[i]/indx[i], label = f' Theoretical for N = {i}')
    plt.legend(loc = 'upper right')

#plt.show()

print("%.2f" % Z5[-1],"%.2f"%Z10[-1],"%.2f"%Z20[-1])