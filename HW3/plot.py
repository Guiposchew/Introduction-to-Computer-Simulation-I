import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# M = pd.read_csv("Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW3/M.csv", sep =';').to_numpy().squeeze()
# E_t = pd.read_csv("Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW3/E_t.csv", sep =';').to_numpy().squeeze()
# E_t_M = pd.read_csv("Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW3/E_t_M.csv", sep =';').to_numpy().squeeze()
# beta = pd.read_csv("Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW3/beta.csv", sep =';').to_numpy().squeeze()

# C = np.gradient(E_t,1/beta)

# plt.figure(1)
# plt.title(r'$<E> & M \times T$')
# plt.xlabel('T')


# plt.plot(1/beta,E_t_M,label= 'Mean energy of the system')

# plt.plot(1/beta, M, label = 'Magnetization of the system')
# plt.legend(loc='lower right')
# plt.show()

N = 10000
A_c = np.pi
A_s = 4

iterations = 1000

error = np.zeros(iterations)
area = np.zeros(iterations)

for i in range(0,iterations):
    r = np.random.rand(2,N)*2-1

    mask = (r[0]**2 + (np.pi*r[1]/2)**2) <1


    p = len(r[0,mask])/N

    area[i] = p*A_s

area_mean = np.mean(area)

print(area_mean,p)



plt.figure(1)
plt.title('Random distribution of numbers in [-1,1)')
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(r[0],r[1], '.', color = 'red', label='Point of coordinate (x,y)')
plt.plot(r[0,mask],r[1,mask], '.', color = 'blue', label=r'Points in a ellipse, $a=1$, $b=2/\pi$')
plt.legend(loc = 'upper right')
plt.show()