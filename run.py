__author__     = "Jose Rosales"
__copyright__  = "TBD"
__credits__    = "TBD"
__license__    = "TBD"
__maintainer__ = "Jose Rosales"
__email__      = "jose.rosales@ai-solutions.com"
__status__     = "Prototype"

__version__ = "$Revision: 1a40d4eaa00b $"

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from DynamicModels import CRTBP
from DynamicModels import CRTBP_Jacobian
from DynamicModels import P_Jacobian
from DynamicModels import planar_CRTBP_DynSys

from Constants import *


planar_CRTBP = planar_CRTBP_DynSys(MU)


time          = 0.0
target_period = 100.0*6.28
period        = 0.0

delta_t = 1e-3

state_vector = np.zeros(DIM, dtype=np.double)
delta_sv     = np.zeros(DIM, dtype=np.double)
D_P          = np.zeros((DIM, DIM), dtype=np.double)

state_x = []
state_y = []

#state_vector[0] = .9
#state_vector[1] = 1e-1
#state_vector[2] = 1.667478e-1
#state_vector[3] = 0e-1

#
state_vector[0] = 1.001  
state_vector[1] = 0.2
state_vector[2] = .4
state_vector[3] =  -0.271 


print planar_CRTBP.get_Jacobi_Constant()


#sets the integrator
r = ode(CRTBP).set_integrator('dopri5')
r.set_initial_value(state_vector, time)


# Generates the Poincare Map
# The section is defined as x=state_vector[0]
#
# we want P(t_f, x(t_f) = x(t_0)
p_section  = state_vector[0] 


#integrates until reaches 'target_period'

for i in range(1,30):

    pos_x = []
    pos_y = []
    vel_x = []
    vel_y = []
    
    pos_x.append(state_vector[0])     
    pos_y.append(state_vector[1])    
    vel_x.append(state_vector[2])     
    vel_y.append(state_vector[3])      
    
    continue_flag = True
    index         = 0
    
    while time < target_period and continue_flag:
        
        planar_CRTBP.set_initial_condition(state_vector)
        planar_CRTBP.set_t0(time)
        planar_CRTBP.set_tf(time+delta_t)
    
        planar_CRTBP.go()
        
        state_vector = planar_CRTBP.get_updated_state_vector()
        time         = planar_CRTBP.get_updated_time()
        
       
        if ((pos_x[-1]-p_section)*(state_vector[0]-p_section)<0.0 and
            (vel_x[0]*state_vector[2] + vel_y[0]*state_vector[3])>0.0):
            index = index + 1
            if index == 1:
                continue_flag = False

        pos_x.append(state_vector[0])     
        pos_y.append(state_vector[1])    
        vel_x.append(state_vector[2])     
        vel_y.append(state_vector[3])
                

    # get period
    period = planar_CRTBP.get_updated_time()
     
    #Solve for delta_sv

    print planar_CRTBP.get_exec_flag(), state_vector

    delta_sv = [pos_x[0], pos_y[0], vel_x[0], vel_y[0]]-state_vector
   
#    D_P =  planar_CRTBP.P_Jacobian(time, state_vector)
    
    D_P =  P_Jacobian(time, state_vector)
    
    
    D_P = np.linalg.inv(D_P)
    
    delta_sv = np.dot(D_P, delta_sv)
    
    state_vector = [pos_x[0], pos_y[0], vel_x[0], vel_y[0]] - delta_sv   
    time = 0.0
    print period, delta_sv, state_vector 
                
    # plotting
    
    plt.annotate('P1', xy=(-MU, 0), xytext=(-MU, -.1))
    plt.annotate('P2', xy=(-MU, 0), xytext=(1-MU, -.1))
    plt.plot(-MU, 0, 'ro')
    plt.plot(1-MU, 0, 'bo')
       
    plt.figure(1)
    plt.plot([p_section, p_section], [-1, 1])
    plt.plot(pos_x, pos_y)
    plt.plot(pos_x[-1], pos_y[-1], 'go')
    
    plt.figure(2)  
    plt.plot(vel_x, vel_y)    

    plt.show()    


##
#target_period = 5.0 * period
#time = 0.0
#
#while time <= target_period:
#
#    
#    planar_CRTBP.set_initial_condition(state_vector)
#    planar_CRTBP.set_t0(time)
#    planar_CRTBP.set_tf(time+delta_t)
#
#    planar_CRTBP.go()
#    
#    state_vector = planar_CRTBP.get_updated_state_vector()
#    time         = planar_CRTBP.get_updated_time()
#    
#    pos_x.append(state_vector[0])     
#    pos_y.append(state_vector[1])    
#    vel_x.append(state_vector[2])     
#    vel_y.append(state_vector[3])
# 
#
#print '**** time *****', time   
##
##    # plotting
##
#    
#plt.annotate('P1', xy=(-MU, 0), xytext=(-MU, -.1))
#plt.annotate('P2', xy=(-MU, 0), xytext=(1-MU, -.1))
#plt.plot(-MU, 0, 'ro')
#plt.plot(1-MU, 0, 'bo')
##
##
#plt.figure(1)
#plt.plot([p_section, p_section], [-1, 1])
#plt.plot(pos_x, pos_y)
##
#plt.figure(2)  
#plt.plot(vel_x, vel_y)    
##
#plt.show()


    
    