__author__     = "Jose Rosales"
__copyright__  = "TBD"
__credits__    = "TBD"
__license__    = "TBD"
__maintainer__ = "Jose Rosales"
__email__      = "pepe.rosales@gmail.com"
__status__     = "Prototype"

import numpy as np

#
# This function returns the seed to initialize the differential
# corrector to compute Lyapunov periodic orbit.
#
# The inputs are:
#   amplitude_: the amplitude of the desired Lyapunov orbit
#   mu_       : the mu of the system
#   L_i_      : x-coordinate of the libration point L_i, (i = 1, 2, 3)
#
# The outputs are:
#   seed_state_vector : a 6-dimensional vector with the initial condition 
#                       (x, y, z, v_x, v_y, v_z)
#

def Lyapunov_seed(amplitude_, mu_, L_i_):
    
    seed_state_vector = np.zeros(6, dtype=np.double)
   
    L_plus_mu = L_i_ + mu_
    mu_bar = L_plus_mu - 1.0
    mu_bar = mu_ / (mu_bar*mu_bar*mu_bar)
    mu_bar = mu_bar + (1.0 - mu_) * 1.0/abs(L_plus_mu*L_plus_mu*L_plus_mu)
    
    nu = (9.0*mu_bar - 8.0)*mu_bar
    nu = -np.sqrt(nu)
    nu = mu_bar - 2.0 + nu
    nu = 0.5*nu
    
    tau = nu*nu + 2.0*mu_bar + 1.0
    tau = -tau/(2.0*nu)
    
    A_x = amplitude_
    v_y = -A_x*nu*tau
    
    seed_state_vector[0] = L_i_ - A_x
    seed_state_vector[1] = 0.0
    seed_state_vector[2] = 0.0
    seed_state_vector[3] = 0.0
    seed_state_vector[4] = v_y
    seed_state_vector[5] = 0.0    
    
    return seed_state_vector 