import math
import time
from collections import defaultdict
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
Nmax = 20


# Allowed lattice steps (square lattice)
steps = [(1,0), (-1,0), (0,1), (0,-1)]


# ------------------------------------------------------------
# Storage for results
# ------------------------------------------------------------
Z = [0]*(Nmax+1) # number of SAWs of length N
R2_sum = [0.0]*(Nmax+1) # sum of end-to-end squared distances


# ------------------------------------------------------------
# Recursive enumeration (depth-first search)
# ------------------------------------------------------------


def enumerate_saw(n, x, y, visited):

    # Count current walk
    Z[n] += 1
    R2_sum[n] += x*x + y*y


    if n == Nmax:
        return


    for dx, dy in steps:
        xn, yn = x + dx, y + dy
    if (xn, yn) not in visited:
        visited.add((xn, yn))
        enumerate_saw(n+1, xn, yn, visited)
        visited.remove((xn, yn))


# ------------------------------------------------------------
# Run enumeration
# ------------------------------------------------------------
start = time.time()


visited = set()
visited.add((0,0))


enumerate_saw(0, 0, 0, visited)


end = time.time()
print(f"Enumeration finished in {end-start:.2f} seconds")


# ------------------------------------------------------------
# Compute mean squared end-to-end distance
# ------------------------------------------------------------
R2_mean = [0.0]*(Nmax+1)
for n in range(Nmax+1):
    if Z[n] > 0:
        R2_mean[n] = R2_sum[n] / Z[n]

print(" N Z(N) <R_ee^2>")
print("---------------------------------")
for n in range(Nmax+1):
    print(f"{n:2d} {Z[n]:12d} {R2_mean[n]:12.6f}")


# ------------------------------------------------------------
# Scaling analysis: <R_ee^2> ~ N^{2 nu}, nu = 3/4 in 2D
# ------------------------------------------------------------
Nvals = [n for n in range(1, Nmax+1)]
lnN = [math.log(n) for n in Nvals]
lnR2 = [math.log(R2_mean[n]) for n in Nvals]
lnZ = [math.log(Z[n]) for n in Nvals]


# ------------------------------------------------------------
# Plot ln <R_ee^2> vs ln N
# ------------------------------------------------------------
plt.figure()
plt.plot(lnN, lnR2, 'o', label=r'$\ln\langle R_{ee}^2\rangle$')


# Reference slope: 2 nu = 3/2
ref = [1.5*x for x in lnN]
plt.plot(lnN, ref, '-', label=r'slope $2\nu = 3/2$')


plt.xlabel(r'$\ln N$')
plt.ylabel(r'$\ln \langle R_{ee}^2 \rangle$')
plt.legend()
plt.title('Scaling of end-to-end distance for 2D SAWs')
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Plot ln Z vs ln N (effective growth is exponential,
# but power-law corrections can be inspected)
# ------------------------------------------------------------
plt.figure()
plt.plot(lnN, lnZ, 'o', label=r'$\ln Z(N)$')
plt.xlabel(r'$\ln N$')
plt.ylabel(r'$\ln Z$')
plt.legend()
plt.title('Number of 2D SAWs')
plt.tight_layout()
plt.show()