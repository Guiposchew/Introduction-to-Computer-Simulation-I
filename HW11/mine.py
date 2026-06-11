import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy import ndimage as nd
import scipy



def get_Random_state(X):
    state = np.random.random(size = X)
    return np.ceil(state*2-1)*2-1

def time_Evolution_Metropolis(state,J,h,beta,iterations, get_Energy_state):
    state_mu = state.copy()
    state_evolution = np.zeros((iterations, len(state)))
    net_Energy = np.zeros(iterations)
    net_Energy[0] = get_Energy_state(state,J,h)

    for i in range(0,iterations):
        state_nu = state_mu.copy()
        x = np.random.randint(0,len(state))
        state_nu[x] = state_nu[x]*-1

        E_nu = get_Energy_state(state_nu,J,h)
        E_mu = get_Energy_state(state_mu,J,h)

        if (E_nu <= E_mu)or(np.random.random()<=np.exp(-beta*(E_nu-E_mu))):
            state_mu = state_nu
            net_Energy[i] = E_nu
        else:
            net_Energy[i] = E_mu

        state_evolution = np.concatenate((state_evolution, state_mu), axis=0)

    return state_evolution, net_Energy, get_Energy_state(state_evolution[-1],J,h)

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
    D = state.ndim
    kern = nd.generate_binary_structure(D,1)
    idx = tuple([1] * D)
    kern[idx] = False
    arr = nd.convolve(state, kern, mode='wrap')

    return (-J*state*arr).sum()/2

N = 200

L = np.array([[4,4],[8,8],[16,16]])
it = np.arange(N)
E_f = np.zeros(N)

beta = np.log(1+np.sqrt(2))/2

state = get_Random_state(L[2])

# for i in range(0,N):
#     _, E_M, _ = time_Evolution_Metropolis(state,1,0, beta, N, get_Energy_state_ND)
#     E_f[i] = get_running_average(E_M)[-1]

A1 = np.zeros(N)
A2 = np.zeros(N)
Ac = np.zeros(N)

beta_c = 0.440686
beta1 = 0.375
beta2 = 0.475

state = get_Random_state(L[2])

for i in range(N):
    state = get_Random_state(L[2])
    State, E_M, E_last = time_Evolution_Metropolis(state,1,0, beta_c, N, get_Energy_state_ND)
    Ac[i] = E_last

for i in range(N):
    state = get_Random_state(L[2])
    State, E_M, E_last = time_Evolution_Metropolis(state,1,0, beta1, N, get_Energy_state_ND)
    A1[i] = E_last

for i in range(N):
    state = get_Random_state(L[2])
    State, E_M, E_last = time_Evolution_Metropolis(state,1,0, beta2, N, get_Energy_state_ND)
    A2[i] = E_last



# hist_1, bin_edges = np.histogram(A1, bins=bins, density=True)
# E_1 = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# hist_c, bin_edges = np.histogram(A2, bins=bins, density=True)
# E_2 = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# hist_1, _ = np.histogram(A1, bins=bin_edges, density=True)
# hist_2, _ = np.histogram(A2, bins=bin_edges, density=True)

# hist_rw_1 = hist_c * np.exp(-(beta1 - beta_c) * E_mid)
# hist_rw_2 = hist_c * np.exp(-(beta2 - beta_c) * E_mid)

# # normalize histograms
# hist_rw_1 /= np.sum(hist_rw_1 * np.diff(bin_edges))
# hist_rw_2 /= np.sum(hist_rw_2 * np.diff(bin_edges))

# bins = bin_edges  # reuse identical bins

# # Direct simulation at beta1
# plt.hist(
#     E_1,
#     bins=bins,
#     density=True,
#     alpha=0.5,
#     label='Direct β1'
# )

# # Reweighted histogram (use weights!)
# plt.hist(
#     E_mid,
#     bins=bins,
#     weights=hist_rw_1 * np.diff(bins),
#     linewidth=2,
#     label='Reweighted β1'
# )

# plt.xlabel("Energy")
# plt.ylabel("P(E)")
# plt.legend()
# plt.title("Energy distribution at β₁ = 0.375")
# plt.show()

bins = 50

# Histogram at beta_c
hist_c, bin_edges = np.histogram(Ac, bins=bins, density=True)
E_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
widths = np.diff(bin_edges)

# Reweighting
P_beta1 = hist_c * np.exp(-(beta1 - beta_c) * E_mid)
P_beta2 = hist_c * np.exp(-(beta2 - beta_c) * E_mid)

# Normalize reweighted histograms
P_beta1 /= np.sum(P_beta1 * widths)
P_beta2 /= np.sum(P_beta2 * widths)

# --- Direct histograms from separate simulations ---
hist_1, _ = np.histogram(A1, bins=bin_edges, density=True)
hist_2, _ = np.histogram(A2, bins=bin_edges, density=True)

# --- Plot ---
plt.figure(figsize=(7,5))

plt.figure(1)
# beta_c
plt.bar(
    E_mid,
    hist_c,
    width=widths,
    alpha=0.4,
    label=r'$\beta_c$ simulation'
)


# direct beta1, beta2
plt.bar(
    E_mid,
    hist_1,
    width=widths,
    alpha=0.3,
    label=r'Direct $\beta_1$'
)

plt.bar(
    E_mid,
    hist_2,
    width=widths,
    alpha=0.3,
    label=r'Direct $\beta_2$'
)

plt.xlabel("Energy")
plt.ylabel("P(E)")
plt.legend()
plt.tight_layout()
plt.xticks([-12,-12,-8,-4,0,4,8,12])

plt.figure(2)
# reweighted beta1, beta2
plt.bar(
    E_mid,
    P_beta1,
    width=widths,
    fill=False,
    edgecolor='red',
    linewidth=2,
    label=r'Reweighted $\beta_1$'
)

plt.bar(
    E_mid,
    P_beta2,
    width=widths,
    fill=False,
    edgecolor='green',
    linewidth=2,
    label=r'Reweighted $\beta_2$'
)
# direct beta1, beta2
plt.bar(
    E_mid,
    hist_1,
    width=widths,
    alpha=0.3,
    label=r'Direct $\beta_1$'
)

plt.bar(
    E_mid,
    hist_2,
    width=widths,
    alpha=0.3,
    label=r'Direct $\beta_2$'
)

plt.xlabel("Energy")
plt.ylabel("P(E)")
plt.legend()
plt.tight_layout()
plt.xticks([-12,-12,-8,-4,0,4,8,12])
plt.show()


# plt.figure(1)
# plt.plot(E_mid, hist_c,label=r'$\beta_c$ simulation')
# plt.plot(E_mid, P_beta1,label='Reweighted β1')
# plt.plot(E_mid, P_beta2, label='Reweighted β2')


# plt.xlabel("Energy")
# plt.ylabel("P(E)")
# plt.legend()
# plt.show()
# # data = pd.DataFrame(E_f)
# # data.to_csv(f"Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW6/Q1/E_f{L[2]}_{beta}.csv")