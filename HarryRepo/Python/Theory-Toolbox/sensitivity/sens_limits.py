## OPM sensor toolbox testing

import numpy as np
import matplotlib.pyplot as plt

#Defining functions

#Spin-projection-noise-limit
def spin_proj_noise(gamma, Rel, N, tao):
    delBsnl = (1/gamma)*np.sqrt(Rel/(N*tao))
    return delBsnl

gamma = 7.2*1e9 #7Hz/nT

#equation 1.3 from book
def optimal_sens(gamma, relconst,V,tao):
    delBopt = (1/gamma)*np.sqrt(relconst/(V*tao))
    return delBopt

relconst = 10e-12 #cm^3/s
V = 1 #cm^3
tao = 1

qq = optimal_sens(gamma,relconst,V,tao)

