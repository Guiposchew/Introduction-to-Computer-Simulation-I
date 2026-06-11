import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp

def get_States(N):
    #Creates a string list with all the possible combinations in binary
    a = [list(bin(i).split('b')[1].zfill(N)) for i in range(2**N)]
    indx = np.zeros((len(a),len(a[0])))

    #Converts the list into an np array with integers
    for i in range(0,len(a)):
        for j in range(0, len(a[0])):
            indx[i][j] = int(a[i][j])
    
    #Transforms the binary array such that 0 -> 1 & 1 -> -1
    return indx*(-2)+1

def get_Energy_all_States(N,J, h=0):
    #set necessary arrays
    state = get_States(N)
    Energy_interaction = np.zeros(len(state))
    Magnetization = np.sum(h*get_States(N), axis = 1)

    for j in range(0,len(state)):
        Energy_j = 0

        #Assigns the energy of state j into the jth element of the Energy array
        for i in range(0,len(state[0])-1):
            Energy_j += state[j][i]*state[j][i+1]

        Energy_interaction[j] = Energy_j

    Energy_total = -J*Energy_interaction - Magnetization
    
    return Energy_total, -J*Energy_interaction , Magnetization

def get_Partition_Function(N, J, beta_max, beta_min = 0.01, step = 0.01):
    #Gets the Energies of all possible states
    E , _, _ = get_Energy_all_States(N, J)

    #All possible beta values in the step resolution (doesnt include zero to avoid problems when computing T = 1/kb*beta )
    beta = np.arange(beta_min, beta_max+step, step)

    #Creates an array with the sum of e^-beta*E for each value of beta
    Z = np.sum(np.exp(-np.outer(beta, E)), axis=1)

    #Returns all the relevant variables
    return Z, E

def get_Probability(E, Z, beta, E_indx):
    #Calculates the degeneracy associated to the energy level E_indx
    degeneracy = np.count_nonzero(E_indx == E)

    #Returns the probability distribution as a function of beta
    return degeneracy*np.exp(-beta*E)/Z

def get_Omega(A):
    #returnts unique values in A and their frequency
    return np.unique(A, return_counts=True)

def get_Mean(A, beta):
    A_unique, Omega = get_Omega(A)

    return np.sum(A_unique*Omega*(np.exp(-np.outer(beta, A_unique))), axis =1)/np.sum(Omega*(np.exp(-np.outer(beta, A_unique))), axis =1)

E_t , E_in, M = get_Energy_all_States(4,1,0)

E_t_u , g = get_Omega(E_t)

print(2*sp.binom(3,(E_t_u-np.min(E_t_u))/2))