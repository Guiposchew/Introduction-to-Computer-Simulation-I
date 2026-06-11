import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy import ndimage as nd
import scipy



def get_Random_state(X):
    state = np.random.random(size = X)
    return np.ceil(state*2-1)*2-1

def time_Evolution_Metropolis(state, J, h, beta, iterations, get_Energy_state):
    state_mu = state.copy()
    net_Energy = np.zeros(iterations)

    for i in range(iterations):
        state_nu = state_mu.copy()

        # pick random lattice site
        x = tuple(np.random.randint(0, s) for s in state.shape)
        state_nu[x] *= -1

        E_mu = get_Energy_state(state_mu, J, h)
        E_nu = get_Energy_state(state_nu, J, h)

        if E_nu <= E_mu or np.random.rand() < np.exp(-beta*(E_nu-E_mu)):
            state_mu = state_nu
            net_Energy[i] = E_nu
        else:
            net_Energy[i] = E_mu

    return state_mu, net_Energy

def get_running_average(X,N_max = 0):
    if N_max == 0:
        N_max = len(X)

    X_r_avg = np.zeros(N_max)

    for i in range(1,N_max):
        X_r_avg[i] = np.sum(X[:i])/i

    return X_r_avg

def time_Evolution_Glauber(state,J,h,beta,iterations, get_Energy_state):
    state_mu = state.copy()
    state_evolution = np.zeros((iterations, len(state)))
    net_Energy = np.zeros(iterations)
    net_Energy[0] = get_Energy_state(state,J,h)

    for i in range(iterations):
        x = tuple(np.random.randint(0, s) for s in state.shape)
        S = state_mu[(x-1)%len(state_mu)] + state_mu[(x+1)%len(state_mu)]
        deltaE = 2*state_mu[x]*S
        p = 1/(1+np.exp(deltaE*beta))

        if (np.random.random()<=p):
            state_mu[x] = -state_mu[x]

        state_evolution = np.concatenate((state_evolution, state_mu), axis=0)
        net_Energy[i] = get_Energy_state(state_mu,J,h)

    return state_evolution, net_Energy, get_Energy_state(state_evolution[-1],J,h)

def get_Energy_state_ND(state,J,h=0):
    D = int((np.array(np.shape(state))/np.shape(state)[0]).sum())
    #D = state.ndim
    kern = nd.generate_binary_structure(D,1)
    idx = tuple([1] * D)
    kern[idx] = False
    arr = nd.convolve(state, kern, mode='wrap')

    return (-J*state*arr).sum()/2

def reweighted_energy(beta, beta0, E_samples):
    w = np.exp(-(beta - beta0) * np.array(E_samples))
    return np.sum(E_samples * w) / np.sum(w)

def reweighted_specific_heat(beta, beta0, E_samples, L):
    E = np.array(E_samples)
    w = np.exp(-(beta - beta0) * E)
    Emean = np.sum(E * w) / np.sum(w)
    E2mean = np.sum(E**2 * w) / np.sum(w)
    return beta**2 / (L**2) * (E2mean - Emean**2)

def metropolis_sweep(state, beta):
    Lx, Ly = state.shape
    for _ in range(Lx * Ly):
        i = np.random.randint(0, Lx)
        j = np.random.randint(0, Ly)

        s = state[i, j]
        nb = (
            state[(i+1)%Lx, j] +
            state[(i-1)%Lx, j] +
            state[i, (j+1)%Ly] +
            state[i, (j-1)%Ly]
        )

        dE = 2 * s * nb

        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            state[i, j] = -s

    return state

def reweight_histogram(H, beta, beta0):
    H_rw = {}
    Z = 0.0
    for E, count in H.items():
        w = count * np.exp(-(beta - beta0) * E)
        H_rw[E] = w
        Z += w
    # normalize to same total counts
    for E in H_rw:
        H_rw[E] *= (sum(H.values()) / Z)
    return H_rw

def observables_from_hist(H, beta, L):
    Z = sum(H.values())
    Emean = sum(E * c for E, c in H.items()) / Z
    E2mean = sum(E**2 * c for E, c in H.items()) / Z
    e = Emean / L**2
    C = beta**2 / L**2 * (E2mean - Emean**2)
    return e, C

def hist_to_arrays(H):
    E = np.array(sorted(H.keys()))
    counts = np.array([H[e] for e in E])
    return E, counts

L = np.array([16,16])

beta_c = 0.440686
beta1 = 0.375
beta2 = 0.475
N_eq = 5000          # equilibration sweeps
N_meas = 20000       # measurement sweeps



state = get_Random_state(L)


state = get_Random_state((16,16))

# equilibration
for _ in tqdm(range(0,2000), desc=f"1"):
    metropolis_sweep(state, beta_c)

# measurements
E_samples = []
for _ in tqdm(range(0,10000), desc=f"2"):
    metropolis_sweep(state, beta_c)
    E_samples.append(get_Energy_state_ND(state, 1))

E1 = reweighted_energy(beta1, beta_c, E_samples) / L[0]**2
C1 = reweighted_specific_heat(beta1, beta_c, E_samples, L[0])

energies = np.array(E_samples)

E_vals, counts = np.unique(energies, return_counts=True)

# This is H_{β0}(E)
H_beta0 = dict(zip(E_vals, counts))

bins = np.arange(-2*16*16, 2*16*16 + 4, 4)

hist, bin_edges = np.histogram(E_samples, bins=bins)

E_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:])

H_beta1 = reweight_histogram(H_beta0, beta1, beta_c)
H_beta2 = reweight_histogram(H_beta0, beta2, beta_c)

E0, H0 = hist_to_arrays(H_beta0)
E1, H1 = hist_to_arrays(H_beta1)
E2, H2 = hist_to_arrays(H_beta2)

P0 = H0 / np.sum(H0)
P1 = H1 / np.sum(H1)
P2 = H2 / np.sum(H2)

print(observables_from_hist(H_beta1, beta1, L),observables_from_hist(H_beta2, beta2, L),observables_from_hist(H_beta0, beta_c, L))

# plt.figure(figsize=(7,5))

# plt.bar(E0, P0, width=4, alpha=0.4, label=r'$\beta_c$')
# plt.bar(E1, P1, width=4, fill=False, edgecolor='red',
#         linewidth=2, label=r'Reweighted $\beta_1$')
# plt.bar(E2, P2, width=4, fill=False, edgecolor='green',
#         linewidth=2, label=r'Reweighted $\beta_2$')

# plt.xlabel("Energy E")
# plt.ylabel("P(E)")
# plt.legend()
# plt.tight_layout()
# plt.show()