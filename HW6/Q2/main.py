import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

it = 10000

r_ee_array = np.zeros(it)
r_gy_array = np.zeros(it)

for i in tqdm(range(0,it), desc=f"Random Walks"):
    N = 75

    phi = 2*np.pi*np.random.random(size = N)
    b = np.array([np.cos(phi),np.sin(phi)]).reshape(2,N)

    r_temp = np.cumsum(b, axis=1)
    r = np.array([np.insert(r_temp[0,:],0, 0),np.insert(r_temp[1,:],0, 0)])

    r_com = np.mean(r,axis=1)
    r_ee = np.mean(r[0,:]*r[1,:])

    r_gy = np.mean((r[0,:] - r_com[0])**2 + (r[1,:] - r_com[1])**2)

    r_ee_array[i] = r_ee
    r_gy_array[i] = r_gy

data1 = pd.DataFrame(r_ee_array)
data1.to_csv(f"Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW6/Q2/r_ee_{N}.csv")
data2 = pd.DataFrame(r_gy_array)
data2.to_csv(f"Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW6/Q2/r_gy_{N}.csv")

# plt.xlim(-10,10)
# plt.ylim(-10,10)
# plt.plot(r[0],r[1])
# plt.quiver(*[0,0], *r_com, angles='xy', scale_units='xy', scale=1)
# plt.show()

