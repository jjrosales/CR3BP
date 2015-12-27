__author__     = "Jose Rosales"
__copyright__  = "TBD"
__credits__    = "TBD"
__license__    = "TBD"
__maintainer__ = "Jose Rosales"
__email__      = "pepe.rosales@gmail.com"
__status__     = "Prototype"

__version__ = "$Revision: 1a40d4eaa00b $"

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from DynamicModels import CRTBP
from DynamicModels import CRTBP_Jacobian
from DynamicModels import P_Jacobian
from DynamicModels import planar_CRTBP_DynSys

from PoincareSections import g_poincare

from Constants import *


planar_CRTBP = planar_CRTBP_DynSys(MU)
g = g_poincare()


time          = 0.0
target_period = 1000.0*6.28
period        = 0.0

delta_t = 1e-3

old_gx   = 0

state_vector = np.zeros(DIM, dtype=np.double)
delta_sv     = np.zeros(DIM, dtype=np.double)
D_P          = np.zeros((DIM, DIM), dtype=np.double)

state_x = []
state_y = []

state_vector[0] = .9
state_vector[1] = 1e-1
state_vector[2] = -1.667478e-1
state_vector[3] = 1e-1

#
#state_vector[0] = 0.8
#state_vector[1] = -0.2
#state_vector[2] = .8
#state_vector[3] =  -0.271 


print planar_CRTBP.get_Jacobi_Constant()


#sets the integrator
r = ode(CRTBP).set_integrator('dopri5')
r.set_initial_value(state_vector, time)


# Generates the Poincare Map
# The section is defined as x=state_vector[0]
#
# we want P(t_f, x(t_f) = x(t_0)
p_section  =  state_vector[0] 

g.set_center(state_vector)
g.set_radius(1.25)

#integrates until reaches 'target_period'

radius = g.get_radius()
x_tol = 1e-9


poincare_y    = []
poincare_ydot = []


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
    
    g.set_x(state_vector)
    g.go()
           
          
#    if (old_gx*g.get_gx()<0.0 and
#        np.linalg.norm(abs(state_vector-g.get_center()))<radius and
#        time > 1.0):
            
    if (old_gx*g.get_gx()<0.0):   

        sv_aux = state_vector
        t_aux  = time
        
        delta = 0.0

        while abs(g.get_gx())>x_tol:
            delta = -g.get_gx()/np.dot(g.get_Dg(), planar_CRTBP.get_f_eval())
            planar_CRTBP.set_initial_condition(sv_aux)
            planar_CRTBP.set_t0(t_aux)
            planar_CRTBP.set_tf(t_aux+delta)

            planar_CRTBP.go()
            
            sv_aux = planar_CRTBP.get_updated_state_vector()
            t_aux  = planar_CRTBP.get_updated_time()

#            print '----->',  delta 

            g.set_x(sv_aux)
            g.go()
        
        poincare_y   .append(state_vector[1])
        poincare_ydot.append(state_vector[3])
        print t_aux, sv_aux
        
#        continue_flag = False

    pos_x.append(state_vector[0])     
    pos_y.append(state_vector[1])    
    vel_x.append(state_vector[2])     
    vel_y.append(state_vector[3])
    
    g.set_x(state_vector)
    g.go()
    old_gx = g.get_gx()
                

    # get period
    period = planar_CRTBP.get_updated_time()
     
    #Solve for delta_sv

#    print planar_CRTBP.get_exec_flag(), state_vector

#    delta_sv = [pos_x[0], pos_y[0], vel_x[0], vel_y[0]]-state_vector
   
#    D_P =  planar_CRTBP.P_Jacobian(time, state_vector)
#    
#    D_P =  P_Jacobian(time, state_vector)
#    
#    
#    D_P = np.linalg.inv(D_P)
#    
#    delta_sv = np.dot(D_P, delta_sv)
#    
#    state_vector = [pos_x[0], pos_y[0], vel_x[0], vel_y[0]] - delta_sv   
#    time = 0.0
#    print period, delta_sv, state_vector 
                
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

plt.figure(3)  
plt.plot(poincare_y, poincare_ydot, '.')   

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


    
    