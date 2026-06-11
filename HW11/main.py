import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW6/Q1/E_f[16 16]_0.csv").to_numpy()

t = data[:, 0]
A = data[:, 1]

N = len(A)

err_naive = np.sqrt(np.var(A, ddof=1) / N)

max_level = int(np.log2(N))
bin_sizes = []
errors = []

A_block = A.copy()

for level in range(max_level):
    n = len(A_block)

    mean = np.mean(A_block)
    var = np.var(A_block, ddof=1)

    err = np.sqrt(var / n)

    bin_sizes.append(2**level)
    errors.append(err)

    if len(A_block) % 2 == 1:
        A_block = A_block[:-1]

    A_block = 0.5 * (A_block[0::2] + A_block[1::2])

bin_sizes = np.array(bin_sizes)
errors = np.array(errors)

err_plateau = errors[-1]
tau_int = 0.5 * (err_plateau / err_naive)**2

print("Naive error:        ", err_naive)
print("Binning plateau:   ", err_plateau)
print("Autocorrelation τ:", tau_int)

plt.figure('Binning_Naive_error_0')
plt.title(r'Naive and Binning error at $\beta$ =  0')
plt.plot(bin_sizes, errors, "-", label="Binning error")
plt.axhline(err_naive, linestyle="--", color = 'orange', label="Naive error")
plt.xscale("log", base=2)
plt.xlabel("bin size")
plt.ylabel("statistical error")
plt.legend()
plt.show()
