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

from Constants import *

np.set_printoptions(linewidth = 300)

#DynSys     = DynamicModels.CRTBP_DynSys(MU)

DynSys = DynamicModels.CRTBP_variational_DynSys(MU)

g = PoincareSections.g_poincare_3d()


DIM           = DynSys.get_dim()
time          = 0.0
target_period = 4*6.28
period        = 0.0

delta_t = 1e-3

old_gx   = 0

# Vectors related to the state vector
ini_sv       = np.zeros(DIM, dtype=np.double)
state_vector = np.zeros(DIM, dtype=np.double)
delta_sv     = np.zeros(DIM, dtype=np.double)
variationals = np.eye(DIM, dtype=np.double)
f_eval       = np.zeros((1, DIM), dtype=np.double)

sv_aux  = state_vector
var_aux = variationals

#Vectors and arrays needed to implement the Newton method to find periodic orbits 
D_P  = np.zeros((DIM, DIM), dtype=np.double)
DgP  = np.zeros((1,   DIM), dtype=np.double)
DgPf = np.zeros((1,   DIM), dtype=np.double)

state_x = []
state_y = []
state_z = []

#state_vector[0] = 0.9
#state_vector[1] = 1e-2
#state_vector[2] = -5e-4
#state_vector[3] = 4e-1
#state_vector[4] = 4.459e-1
#state_vector[5] = -1.2e-6

#
state_vector[0] = 0.994
state_vector[1] = 0.0
state_vector[2] = 1e-3
state_vector[3] = 0.0
state_vector[4] = -2.0317326295573368357302057924
state_vector[5] = 0.0

#state_vector[0:3] = DynSys.get_libration_points()[0]

p = [0.0, -0.02, 0.0]#state_vector

p_section = p[1]


g.set_center(p)
g.set_radius(1.25)

#integrates until reaches 'target_period'

radius = g.get_radius()
x_tol = 1e-9


poincare_y    = []
poincare_ydot = []


pos_x = []
pos_y = []
pos_z = []
vel_x = []
vel_y = []
vel_z = []

pos_x.append(state_vector[0])     
pos_y.append(state_vector[1])   
pos_z.append(state_vector[2])   
vel_x.append(state_vector[3])     
vel_y.append(state_vector[4])      
vel_z.append(state_vector[5])      

continue_flag = True
index         = 0


DynSys.set_initial_condition (state_vector)
DynSys.set_variationals  (variationals)    

print state_vector
print DynSys.get_Jacobi_Constant()

for i in range (0,20):
    
    ini_sv = state_vector
    continue_flag = True    
    
    print '+++>', ini_sv
    
    while time < target_period and continue_flag:
      
        DynSys.set_initial_condition (state_vector)
        DynSys.set_variationals      (variationals)    
        DynSys.set_t0                (time)
        DynSys.set_tf                (time+delta_t)
    
        DynSys.go()
        
        state_vector = DynSys.get_updated_state_vector()
        variationals = DynSys.get_updated_variationals() 
        time         = DynSys.get_updated_time()

        pos_x.append(state_vector[0])     
        pos_y.append(state_vector[1])   
        pos_z.append(state_vector[2])   
        vel_x.append(state_vector[3])     
        vel_y.append(state_vector[4])      
        vel_z.append(state_vector[5])  
    
        f_eval[0,0:DIM] = DynSys.get_f_eval(state_vector)
     
        DynSys.reset()
    
        g.set_x(state_vector)
        g.go()
                                    
        if (old_gx*g.get_gx()<0.0):   
    
            sv_aux  = state_vector
            var_aux = variationals
            t_aux   = time
    
            index = index + 1
          
            delta = 0.0
    
            while abs(g.get_gx())>x_tol:
                
                delta = (-g.get_gx()/np.dot(g.get_Dg(), np.transpose(f_eval)))[0]
                
                DynSys.set_initial_condition(sv_aux)
                DynSys.set_variationals(var_aux)
                DynSys.set_t0(t_aux)
                DynSys.set_tf(t_aux+delta)
    
                DynSys.go()
                
                sv_aux  = DynSys.get_updated_state_vector()
                var_aux = DynSys.get_updated_variationals() 
                t_aux   = DynSys.get_updated_time()
    
                f_eval[0,0:DIM] =  DynSys.get_f_eval(sv_aux)            
                
                DynSys.reset()
    
                g.set_x(sv_aux)
                g.go()

                if index == 4:
                    index = 0
                    continue_flag = False
                    
            poincare_y   .append(sv_aux[1])
            poincare_ydot.append(sv_aux[2])
            print '--> ', t_aux, sv_aux
            
        if continue_flag == False:
            DgPf[0, 0:DIM] =  np.dot(np.transpose(g.get_Dg()),var_aux)/np.dot(g.get_Dg(), np.transpose(f_eval))
            D_P  = -np.dot(np.transpose(f_eval), DgPf)
    #        print '----------------------------'
    #        print np.dot(g.get_Dg(), np.transpose(f_eval))
    #        print
    #        print DgPf
    #        print
    #        print D_P
            D_P = D_P + var_aux  
            print
            print D_P
            #Solve for delta_sv
            D_P = np.linalg.inv(D_P)
            
            delta_sv = -(sv_aux - ini_sv)
            
            delta_sv = np.dot(D_P, delta_sv)
            print
            print delta_sv
     
            state_vector = ini_sv - delta_sv
            variationals = np.eye(DIM, dtype=np.double)
            time = 0.0

        g.set_x(state_vector)
        g.go()                    
        old_gx = g.get_gx()                
       

                    
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
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot([-MU], [0.0], [0.0], 'ro')
    ax.plot([1-MU],[0.0], [0.0], 'bo')
    ax.plot(pos_x, pos_y, pos_z, 'g')
    
    plt.show()    


    pos_x = []
    pos_y = []
    pos_z = []  
    vel_x = []    
    vel_y = []      
    vel_z = []

##
#target_period = 5.0 * period
#time = 0.0
#
#while time <= target_period:
#
#    
#    DynSys.set_initial_condition(state_vector)
#    DynSys.set_t0(time)
#    DynSys.set_tf(time+delta_t)
#
#    DynSys.go()
#    
#    state_vector = DynSys.get_updated_state_vector()
#    time         = DynSys.get_updated_time()
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


    
    