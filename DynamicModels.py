__author__     = "Jose Rosales"
__copyright__  = "TBD"
__credits__    = "TBD"
__license__    = "TBD"
__maintainer__ = "Jose Rosales"
__email__      = "jose.rosales@ai-solutions.com"
__status__     = "Prototype"

__version__ = "$Revision: 1a40d4eaa00b $"

# $Source$

import numpy as np
from   Constants import *


def CRTBP (time, state_vector):
    
    state_update = np.zeros(DIM, dtype=np.double)
    
    one_minus_mu = 1.0 - MU
    
    r2 = state_vector[1]*state_vector[1]
    r2 = r2 + (state_vector[0]-one_minus_mu)*(state_vector[0]-one_minus_mu)
    r2 = r2*r2*r2
    r2 = np.sqrt(r2)
    
    r1 = state_vector[1]*state_vector[1]
    r1 = r1 + (state_vector[0]+MU)*(state_vector[0]+MU)
    r1 = r1*r1*r1
    r1 = np.sqrt(r1)
 
    state_update[0] = state_vector[2] 
    
    state_update[1] = state_vector[3]
    
    state_update[2] = 2.0*state_vector[3] + state_vector[0]
    state_update[2] = state_update[2] - MU*(state_vector[0]-one_minus_mu)/r2
    state_update[2] = state_update[2] - one_minus_mu*(state_vector[0]+MU)/r1 
    
    state_update[3] =-2.0*state_vector[2] + state_vector[1]
    state_update[3] = state_update[3] - MU*(state_vector[1])/r2
    state_update[3] = state_update[3] - one_minus_mu*state_vector[1]/r1 

    return state_update
    
    
def CRTBP_Jacobian(time, state_vector):

    J_CRTBP = np.zeros((DIM, DIM), dtype=np.double)

    one_minus_mu = 1.0 - MU

    r2 = state_vector[1]*state_vector[1]
    r2 = r2 + (state_vector[0]-one_minus_mu)*(state_vector[0]-one_minus_mu)
    r2 = np.sqrt(r2)
    
    r1 = state_vector[1]*state_vector[1]
    r1 = r1 + (state_vector[0]+MU)*(state_vector[0]+MU)
    r1 = np.sqrt(r1)
    
    inv_r1_3 = 1.0/(r1*r1*r1) 
    inv_r1_5 = inv_r1_3/(r1*r1)

    inv_r2_3 = 1.0/(r2*r2*r2) 
    inv_r2_5 = inv_r2_3/(r2*r2)

    
    J_CRTBP[0,2] = 1.0
    J_CRTBP[1,3] = 1.0

    J_CRTBP[2,3] = 2.0
    J_CRTBP[3,2] = -2.0
    
    J_CRTBP[2,0] = one_minus_mu*(-inv_r1_3 + 3.0*(state_vector[0]+MU)*(state_vector[0]+MU)*inv_r1_5)
    J_CRTBP[2,0] = J_CRTBP[2,0] + MU*(-inv_r2_3 + 3.0*(state_vector[0]-one_minus_mu)*(state_vector[0]-one_minus_mu)*inv_r2_5)
    J_CRTBP[2,0] = J_CRTBP[2,0] + 1.0    
    
    J_CRTBP[2,1] = 3.0*one_minus_mu*(state_vector[0]+MU)*state_vector[1]*inv_r2_5
    J_CRTBP[2,1] = J_CRTBP[2,1] + 3.0*MU*(state_vector[0]-one_minus_mu)*state_vector[1]*inv_r1_5

    J_CRTBP[3,0] = J_CRTBP[2,1]
    
    J_CRTBP[3,1] = one_minus_mu*(-inv_r1_3 + 3.0*state_vector[1]*state_vector[1]*inv_r1_5)
    J_CRTBP[3,1] = J_CRTBP[3,1] + MU*(-inv_r2_3 + 3.0*state_vector[1]*state_vector[1]*inv_r2_5)
    J_CRTBP[3,1] = J_CRTBP[3,1] + 1.0       
 
    return J_CRTBP
    
 # returns Jacobian of the Poincare matrix  
def P_Jacobian(time, state_vector):
    
    J_P = np.zeros((DIM, DIM), dtype=np.double)

    J_P = P_Phi(time, state_vector)

    return J_P

# return the first order approximation of the transition matrix    
def P_Phi(time, state_vector):
    
    phi = np.eye(DIM, dtype=np.double)
    phi = phi + time * CRTBP_Jacobian(time, state_vector)  
    
    return phi