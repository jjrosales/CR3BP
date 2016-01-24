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
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode
import DynamicModels
import PoincareSections
from Lyapunov_seed import * 
import planar_lyapunov as planar_lyapunov 
from Constants import *

np.set_printoptions(linewidth = 110)

DynSys = DynamicModels.CRTBP_DynSys(MU)
g = PoincareSections.g_poincare_3d()

DIM           = DynSys.get_dim()
time          = 0.0
target_period = 6.28
period        = 0.0

delta_t  = 1e-3
d        = 1e-12
delta_vy = 1.0

old_gx   = 0

ini_state_vector = np.zeros(DIM, dtype=np.double)
state_vector     = np.zeros(DIM, dtype=np.double)
pos_vel          = np.zeros(6, dtype=np.double)
var              = np.eye(6, dtype=np.double)
old_var          = np.eye(6, dtype=np.double)
delta_sv         = np.zeros(DIM, dtype=np.double)
D_P              = np.zeros((DIM, DIM), dtype=np.double)

i = 1

L_i = DynSys.get_libration_points()[i][0]

planar_lyapunov_po =  planar_lyapunov.planar_lyapunov(DynSys, i+1)

ini_state_vector[0]  = L_i
ini_state_vector[1]  = 0.0
ini_state_vector[2]  = 0.0
ini_state_vector[3]  = 0.0
ini_state_vector[4]  = 0.0 
ini_state_vector[5]  = 0.0
ini_state_vector[6]  = 1.0
ini_state_vector[13] = 1.0
ini_state_vector[20] = 1.0
ini_state_vector[27] = 1.0
ini_state_vector[34] = 1.0
ini_state_vector[41] = 1.0


p = state_vector

p_section = 0.0*p

g.set_center(p_section)
g.set_radius(1.25)

x_tol = 1e-12

poincare_y    = []
poincare_ydot = []
   
index         = 0

# Given an amplitude, computes the seed to initialize the differential
# corrector to compute Lyapunov periodic orbits

#
#ini_state_vector[0:6] = Lyapunov_seed(1e-4, MU, L_i)
#
#DynSys.set_initial_condition(ini_state_vector)
#print DynSys.get_Jacobi_Constant()
#
#while abs(delta_vy)>d:
#    
#    index = index + 1
#
#    continue_flag = True
#
#    old_gx       = 0.0
#    delta_vy     = 0.0  
#    time         = 0.0
#    state_vector = ini_state_vector
#    
#
#    while abs(time) < target_period and continue_flag:
#    
#        # Integrate the equations of motion and the variational equations
#
#        DynSys.set_initial_condition(state_vector)
#        DynSys.set_t0(time)
#        DynSys.set_tf(time+delta_t)
#    
#        DynSys.go()
#        
#        state_vector = DynSys.get_updated_state_vector()
#        time         = DynSys.get_updated_time()
#        
#        pos_vel = DynSys.get_updated_pos_vel()
#        var     = DynSys.get_updated_var()
#    
#        time = time + delta_t
#           
#        g.set_x(state_vector)
#        g.go()
#   
#        # Poincare Map computation
#                
#        if (old_gx*g.get_gx()<0.0):   
#    
#            sv_aux = state_vector
#            t_aux  = time
#            
#            delta = 0.0
#    
#            while abs(g.get_gx())>x_tol:
#                delta = -g.get_gx()/np.dot(g.get_Dg(), DynSys.get_f_eval())
#                DynSys.set_initial_condition(sv_aux)
#                DynSys.set_t0(t_aux)
#                DynSys.set_tf(t_aux+delta)
#    
#                DynSys.go()
#                
#                sv_aux = DynSys.get_updated_state_vector()
#                t_aux  = DynSys.get_updated_time()
#    
#                g.set_x(sv_aux)
#                g.go()
#                
#            half_T = t_aux
#            
#            poincare_y   .append(state_vector[1])
#            poincare_ydot.append(state_vector[2])
#           
#            vdot_y  = DynSys.get_f_eval()[4]
#            pos_vel = DynSys.get_updated_pos_vel()
#            var     = DynSys.get_updated_var()
#
#            delta_vy = (var[3,4] - (var[1,4]*vdot_y)/pos_vel[4]) 
#            delta_vy = pos_vel[3]/delta_vy
#            
#            ini_state_vector[4] = ini_state_vector[4]-delta_vy
#
#            print index, delta_vy, half_T, sv_aux[0:6]
#            
#            continue_flag = False
#    
#    
#        
#        g.set_x(state_vector)
#        g.go()
#        old_gx = g.get_gx()
        

planar_lyapunov_po.set_amplitude(1e-4) 
planar_lyapunov_po.go()
ini_state_vector[0:6] = planar_lyapunov_po.get_state_ini()
ini_state_vector[6]   = 1.0
ini_state_vector[13]  = 1.0
ini_state_vector[20]  = 1.0
ini_state_vector[27]  = 1.0
ini_state_vector[34]  = 1.0
ini_state_vector[41]  = 1.0
        
T_period = planar_lyapunov_po.get_period()

pos_x = []
pos_y = []
pos_z = []
vel_x = []
vel_y = []
vel_z = []

state_vector = ini_state_vector
time         = 0.0

while abs(time) <= T_period:       

    pos_x.append(state_vector[0])     
    pos_y.append(state_vector[1])   
    pos_z.append(state_vector[2])   
    vel_x.append(state_vector[3])     
    vel_y.append(state_vector[4])      
    vel_z.append(state_vector[5])  
  
    DynSys.set_initial_condition(state_vector)
    DynSys.set_t0(time)
    DynSys.set_tf(time+delta_t)

    DynSys.go()
    
    state_vector = DynSys.get_updated_state_vector()
    time         = DynSys.get_updated_time()

#    print time, state_vector

plt.figure(1)
plt.plot(L_i, 0, 'r*')
plt.plot(1-MU, 0, 'bo')
plt.plot(pos_x, pos_y)
    
#                
#    # get period
#    period = DynSys.get_updated_time()
     
    #Solve for delta_sv

#    print DynSys.get_exec_flag(), state_vector

#    delta_sv = [pos_x[0], pos_y[0], vel_x[0], vel_y[0]]-state_vector
   
#    D_P =  DynSys.P_Jacobian(time, state_vector)
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
    
#plt.annotate('P1', xy=(-MU, 0), xytext=(-MU, -.1))
#plt.annotate('P2', xy=(-MU, 0), xytext=(1-MU, -.1))
#plt.plot(-MU, 0, 'ro')
#plt.plot(1-MU, 0, 'bo')
#plt.plot(DynSys.get_libration_points()[0][0], 0, 'r*')
#plt.plot(DynSys.get_libration_points()[1][0], 0, 'r*')
#plt.plot(DynSys.get_libration_points()[2][0], 0, 'r*')
   
#plt.figure(1)
##plt.plot([p_section, p_section], [-1, 1])
#plt.plot(L_i, 0, 'r*')
#plt.plot(pos_x, pos_y)
##plt.plot(pos_x[-1], pos_y[-1], 'go')
#
#plt.figure(2)  
#plt.plot(1-MU, 0, 'bo')
#plt.plot(pos_x, pos_z)    
#
#plt.figure(3)  
#plt.plot(poincare_y, poincare_ydot, '.')   
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
##ax.plot([-MU], [0.0], [0.0], 'ro')
#ax.plot([1-MU],[0.0], [0.0], 'bo')
#ax.plot(pos_x, pos_y, pos_z, 'g')
#
#plt.show()    

