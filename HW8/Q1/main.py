import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

def Gaussian(x, mu, sigma):
    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x - mu)**2 / (2 * sigma**2))


N = [10,20,100]
p = [0.3,0.5,0.7,0.9]
N_s = 100000
Nr = np.zeros(N_s)
R = np.zeros(N_s)

for q in range(len(p)):
    for l in range(len(N)):
        for j in tqdm(range(0,N_s), desc=f"Random Walks"):
            x = np.zeros(N[2]+1)
            for i in range(1,N[2]+1):

                if np.random.rand()<= p[3]:
                    x[i] += 1
                else:
                    x[i] -= 1
            
            r = np.zeros(len(x))

            for k in range(len(x)):
                r[k] = r[k-1] + x[k]

            R[j] = r[-1]
            

        num_distinct = len(np.unique(R))

        y = np.linspace(-len(R)/2,len(R)/2,len(R))

        counts, bin_edges = np.histogram(R, bins=num_distinct, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])


        args, cov = curve_fit(
            Gaussian,
            bin_centers,
            counts
        )


        G = Gaussian(bin_centers, *args)

        mask = G > 0

        ratio = counts[mask] / G[mask]

        plt.figure(f'fit_p{p[3]}_N{N[2]}')
        plt.title(f'Histogram of the final position after N = {N[2]} steps, and p = {p[3]}')
        plt.xlabel('Final position')
        plt.ylabel('Density')
        plt.xlim(-args[1]*4+args[0],args[1]*4+args[0])
        plt.hist(R,density=True,bins=num_distinct)
        plt.plot(y, Gaussian(y,*args), label = f'$\mu$ = {"%.2f"%args[0]}, $\sigma$ = {"%.2f"%args[1]}')
        plt.legend(loc = 'upper right')

        plt.figure(f'Ratio_p{p[3]}_N{N[2]}')
        plt.title(f'Ratio of results and fitted Gaussian, N = {N[2]} steps, and p = {p[3]}')
        plt.xlabel('Final position')
        plt.ylabel('Density')
        plt.plot(bin_centers[mask],ratio)
        plt.show()

       



# N_s = 100000
# Nr = np.zeros(N_s)
# R = np.zeros(N_s)
# N = 500

# p = 0.5

# x = np.zeros(N+1)
# r = np.zeros(len(x))

# k = 0 
# omega = 5

# for i in range(1,N+1):
#     a = np.random.rand()
     
#     if k>=0:
#         if a <= p/(np.abs(k)*omega+1):
#             r[i] = r[i-1]+ 1
#         else:
#             r[i] = r[i-1]- 1
#     else:
#         if a >= p/(np.abs(k)*omega+1):
#             r[i] = r[i-1]+ 1
#         else:
#             r[i] = r[i-1]- 1
        


#     k = r[i]






# y = np.arange(len(r))

# plt.title(r'Position of the random walk where p = $\frac{p_0}{(x\omega + 1)}$')
# plt.xlabel('Position (x)')
# plt.ylabel('Iteration')
# plt.plot(r,y, label = r'$p_0 = 0.5, \omega = 5$')
# plt.legend()
# plt.show()

y = np.arange(len(x))


plt.figure('100steps')
plt.title('1D random walk with 100 steps')
plt.xlabel('Position')
plt.ylabel('Step')
plt.plot(r,y)

plt.figure('10steps')
plt.title('1D random walk with 10 steps')
plt.xlabel('Position')
plt.ylabel('Step')
plt.plot(r[:10],y[:10])

plt.figure('20steps')
plt.title('1D random walk with 20 steps')
plt.xlabel('Position')
plt.ylabel('Step')
plt.plot(r[:20],y[:20])

plt.show()