import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

indx = [10,20,30,40,50,75,100]

r_ee = np.zeros(len(indx))

for i in range(0,len(indx)):
    r_ee[i] = np.mean(pd.read_csv(f"Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW6/Q2/r_ee_{indx[i]}.csv").to_numpy()[:,1])

r_gy = np.zeros(len(indx))

for i in range(0,len(indx)):
    r_gy[i] = np.mean(pd.read_csv(f"Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW6/Q2/r_gy_{indx[i]}.csv").to_numpy()[:,1])

plt.title(f'Values for number of steps: {indx}')
plt.xlabel('Numbers of steps')
plt.ylabel('Numerical Value')
plt.plot(indx,r_ee, label = 'Average squared end-to-end distance')
plt.plot(indx,r_gy, label = 'Average squared radius of gyration')
plt.legend(loc = 'upper left')
plt.show()