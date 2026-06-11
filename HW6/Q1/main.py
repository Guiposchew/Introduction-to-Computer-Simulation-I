import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy import ndimage as nd
import numba 



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
    kern = nd.generate_binary_structure(D,1)
    idx = tuple([1] * D)
    kern[idx] = False
    arr = nd.convolve(state, kern, mode='wrap')

    return (-J*state*arr).sum()/2

N = 200

L = np.array([[4,4],[8,8],[16,16]])
it = np.arange(N)
E_f = np.zeros(N)

beta = 2

state = get_Random_state(L[2])

for i in range(0,N):
    _, E_M, _ = time_Evolution_Metropolis(state,1,0, beta, N, get_Energy_state_ND)
    E_f[i] = get_running_average(E_M)[-1]


data = pd.DataFrame(E_f)
data.to_csv(f"Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW6/Q1/E_f{L[2]}_{beta}.csv")