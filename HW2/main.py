import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def u(N,j,T):
    beta_j = np.linspace(0,T,1000)*j
    return np.array([beta_j, (1/N -1)*j*np.tanh(beta_j)])

def c(N,j,T):
    beta_j = np.linspace(0,T,1000)*j
    return np.array([beta_j, (1- (1/N))*((beta_j/np.cosh(beta_j))**2)])

def get_states(N):
    #Creates a string list with all the possible combinations in binary
    a = [list(bin(i).split('b')[1].zfill(N)) for i in range(2**N)]
    indx = np.zeros((len(a),len(a[0])))

    #Converts the list into an np array with integers
    for i in range(0,len(a)):
        for j in range(0, len(a[0])):
            indx[i][j] = int(a[i][j])
    
    #Transforms the binary array such that 0 -> 1 & 1 -> -1
    return indx*(-2)+1

def get_energy_all_states(N,J):
    #set necessary arrays
    state = get_states(N)
    Energy = np.zeros(len(state))

    for j in range(0,len(state)):
        Energy_j = 0

        #Assigns the energy of state j into the jth element of the Energy array
        for i in range(0,len(state[0])-1):
            Energy_j += state[j][i]*state[j][i+1]

        Energy[j] = Energy_j

    #Multiplies all the energies by -J , also returns all the unique energies for all possible states
    return -J*Energy , np.unique(-J*Energy)

def get_Partition_Function(N, J, beta_max, step = 0.01):
    #Gets the Energies of all possible states
    E , E_unique = get_energy_all_states(N, J)

    #All possible beta values in the step resolution (doesnt include zero to avoid problems when computing T = 1/kb*beta )
    beta = np.arange(step, beta_max+step, step)

    #Creates an array with the sum of e^-beta*E for each value of beta
    Z = np.sum(np.exp(-np.outer(beta, E)), axis=1)

    #Returns all the relevant variables
    return Z , beta, E, E_unique

def get_Probability(E, Z, beta, E_indx):
    #Calculates the degeneracy associated to the energy level E_indx
    degeneracy = np.count_nonzero(E_indx == E)

    #Returns the probability distribution as a function of beta
    return degeneracy*np.exp(-beta*E)/Z

def correction_Free(N,beta):
    return np.log(4*(np.cosh(beta)**((2*N-1)/N)))

def correction_Periodic(N,beta):
    return np.log(4*np.cosh(beta))+(1/N)*np.log(np.cosh(beta)**N + np.sinh(beta)**N)

N = np.arange(1,100,1)
beta = [1,2,10]

f_f = [correction_Free(N,beta[0]),correction_Free(N,beta[1]),correction_Free(N,beta[2])]
f_p = [correction_Periodic(N,beta[0]),correction_Periodic(N,beta[1]),correction_Periodic(N,beta[2])]

for i in range(len(beta)):
    plt.figure(1)
    plt.title(r'$\Delta \beta f$ as a function of N ' '\n' r'in free Boundary conditions')
    plt.xlabel('N')
    plt.ylabel(r'$\Delta \beta f$')
    plt.yscale('log')
    plt.plot(N,f_f[i], label = r'$\Delta \beta f$ for $\beta$ = 'f'{beta[i]}')
    plt.legend(loc = 'lower right')

for i in range(len(beta)):
    plt.figure(2)
    plt.title(r'$\Delta \beta f$ as a function of N ' '\n' r'in periodic Boundary conditions')
    plt.xlabel('N')
    plt.ylabel(r'$\Delta \beta f$')
    plt.yscale('log')
    plt.plot(N,f_p[i], label = r'$\Delta \beta f$ for $\beta$ = 'f'{beta[i]}')
    plt.legend(loc = 'lower right') 



plt.show()