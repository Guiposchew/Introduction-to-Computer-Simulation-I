import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def get_Random_state(N):
    state = np.random.random(N)
    return np.ceil(state*2-1)*2-1

def get_Energy_state(state,J, h=0):
    #set necessary arrays
    Energy_interaction = np.zeros(len(state))
    Magnetization = np.sum(h*state)

    #Assigns the energy of state j into the jth element of the Energy array
    for i in range(0,len(state)-1):
        Energy_interaction[i] = state[i]*state[i+1]

    Energy_total = -J*Energy_interaction.sum() - Magnetization
    
    return Energy_total, -J*Energy_interaction.sum() , Magnetization

def time_Evolution(state,J,h,beta,iterations):

    state_mu = state.copy()
    state_evolution = np.zeros((iterations, len(state)))
    net_Energy = np.zeros(iterations)
    net_Energy[0] = get_Energy_state(state,J,h)[0]

    for i in range(0,iterations):
        state_nu = state_mu.copy()
        x = np.random.randint(0,len(state))
        state_nu[x] = state_nu[x]*-1

        E_nu = get_Energy_state(state_nu,J,h)[0]
        E_mu = get_Energy_state(state_mu,J,h)[0]

        if (E_nu < E_mu)or(np.random.random()<np.exp(-beta*(E_nu-E_mu))):
            state_mu = state_nu
            net_Energy[i] = E_nu
        else:
            net_Energy[i] = E_mu

        state_evolution[i] = state_mu

    return state_evolution, net_Energy, get_Energy_state(state_evolution[-1],J,h)

def get_running_average(X,N_max):
    X_r_avg = np.zeros(N_max)

    for i in range(1,N_max):
        X_r_avg[i] = np.sum(X[:i])/i

    return X_r_avg

beta = 0
J = 1
h = 1

M_all = np.zeros(200)

for i in tqdm(range(0,200), desc="Writing Magnetization in function of Beta"):
    state1 = get_Random_state(20)
    s_1 , E_1, R = time_Evolution(state1,J,h,beta+i/100,10000)
    M_all[i] = R[0]


data = pd.DataFrame(M_all)
data.to_csv("Wi 2025-2026/Introduction to Computer Simulation I/Answers/HW4/Q2/Data/M.csv")