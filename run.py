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

np.set_printoptions(linewidth = 110)

DynSys = DynamicModels.CRTBP_DynSys(MU)
g = PoincareSections.g_poincare_3d()


DIM           = DynSys.get_dim()
time          = 0.0
target_period = 4*6.28
period        = 0.0

delta_t = 1e-3
gamma = 1e-3

old_gx   = 0

state_vector = np.zeros(DIM, dtype=np.double)
pos_vel      = np.zeros(6, dtype=np.double)
var          = np.eye(6, dtype=np.double)
old_var      = np.eye(6, dtype=np.double)
delta_sv     = np.zeros(DIM, dtype=np.double)
D_P          = np.zeros((DIM, DIM), dtype=np.double)

state_x = []
state_y = []
state_z = []

#state_vector[0] = 0.71
#state_vector[1] = 1e-2
#state_vector[2] = -5e-4
#state_vector[3] = 4e-1
#state_vector[4] = 4.459e-1
#state_vector[5] = -1.2e-6

#
#state_vector[0]  = 0.1001005021494284e1
#state_vector[4]  = 0.1215976572734674e-2

#state_vector[0]  = 0.994
#state_vector[1]  = 0.0
#state_vector[2]  = 0.0
#state_vector[3]  = 0.0
#state_vector[4]  = -2.0317326295573368357302057924
#state_vector[5]  = 0.0

#3.23879125e-001 +0.00000000e+00j,  -3.23879125e-001 +0.00000000e+00j,
#         -1.27850912e-001 -4.66300886e-18j,  -1.27850912e-001 +4.66300886e-18j,
#         -2.05063327e-043 -2.22976781e-59j,  -2.05063327e-043 +2.22976781e-59j

#delta_x = 1e-3*[np.cos()]

j = 0

state_vector[0]  = DynSys.get_libration_points()[j][0]
state_vector[1]  = 0.0
state_vector[2]  = 0.0
state_vector[3]  = 0.0
state_vector[4]  = 0.0 #-2.0317326295573368357302057924
state_vector[5]  = 0.0
state_vector[6]  = 1.0
state_vector[13] = 1.0
state_vector[20] = 1.0
state_vector[27] = 1.0
state_vector[34] = 1.0
state_vector[41] = 1.0


p = state_vector

p_section = 0.0*p

g.set_center(p_section)
g.set_radius(1.25)


#integrates until reaches 'target_period'

radius = g.get_radius()
x_tol = 1e-12


poincare_y    = []
poincare_ydot = []


pos_x = []
pos_y = []
pos_z = []
vel_x = []
vel_y = []
vel_z = []

#pos_x.append(state_vector[0])     
#pos_y.append(state_vector[1])   
#pos_z.append(state_vector[2])   
#vel_x.append(state_vector[3])     
#vel_y.append(state_vector[4])      
#vel_z.append(state_vector[5])      

continue_flag = True
index         = 0

print state_vector

DynSys.set_initial_condition(state_vector)

print DynSys.get_Jacobi_Constant()
print DynSys.get_jacobian(state_vector)

#for i in range(0,6):
i = 2
lambda_ = np.linalg.eig(DynSys.get_jacobian(state_vector))[0]
eigenv  = np.linalg.eig(DynSys.get_jacobian(state_vector))[1]
eigenv  = np.transpose(eigenv)

print eigenv

u1 = eigenv[0]
u1 = u1/u1[0]

u2 = eigenv[1]
u2 = u2/u2[0]

w1 = eigenv[2]
w1 = w1/w1[0]

w2 = eigenv[3]
w2 = w2/w2[0]

#print u1
#print u2
#print w1.real
#print w2.real

beta = .001

state_vector[0:6] = 2.0*beta*w1.real
state_vector[0] = state_vector[0] + DynSys.get_libration_points()[j][0]
state_vector[1] = 0.0
state_vector[2] = 0.0
state_vector[3] = 0.0
state_vector[5] = 0.0

print state_vector

DynSys.set_initial_condition(state_vector)
print DynSys.get_Jacobi_Constant()

target_period = 5000*delta_t

while abs(time) < target_period and continue_flag:
  
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
    
    pos_vel = DynSys.get_updated_pos_vel()
    var     = DynSys.get_updated_var()
#    
#    d = var - old_var

#    print time
#    print state_vector
    time = time + delta_t

    print time, state_vector

##    
#    print DynSys.get_jacobian(state_vector)
#
#    print np.linalg.eig(DynSys.get_jacobian(state_vector))

#    print var
#    print old_var
#    print d
#    print '--->', np.linalg.det(var)
#    
#    old_var = var
 
#    print DynSys.get_Jacobi_Constant()
   
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
            delta = -g.get_gx()/np.dot(g.get_Dg(), DynSys.get_f_eval())
            DynSys.set_initial_condition(sv_aux)
            DynSys.set_t0(t_aux)
            DynSys.set_tf(t_aux+delta)

            DynSys.go()
            
            sv_aux = DynSys.get_updated_state_vector()
            t_aux  = DynSys.get_updated_time()

            g.set_x(sv_aux)
            g.go()
        
        poincare_y   .append(state_vector[1])
        poincare_ydot.append(state_vector[2])
        print t_aux, sv_aux

        pos_x.append(sv_aux[0])     
        pos_y.append(sv_aux[1])   
        pos_z.append(sv_aux[2])   
        vel_x.append(sv_aux[3])     
        vel_y.append(sv_aux[4])      
        vel_z.append(sv_aux[5])
        
        continue_flag = False


    
#    g.set_x(state_vector)
#    g.go()
    old_gx = g.get_gx()
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
    
plt.annotate('P1', xy=(-MU, 0), xytext=(-MU, -.1))
plt.annotate('P2', xy=(-MU, 0), xytext=(1-MU, -.1))
plt.plot(-MU, 0, 'ro')
plt.plot(1-MU, 0, 'bo')
plt.plot(DynSys.get_libration_points()[0][0], 0, 'r*')
plt.plot(DynSys.get_libration_points()[1][0], 0, 'r*')
   
plt.figure(1)
#plt.plot([p_section, p_section], [-1, 1])
plt.plot(pos_x, pos_y)
#plt.plot(pos_x[-1], pos_y[-1], 'go')

plt.figure(2)  
plt.plot(1-MU, 0, 'bo')
plt.plot(pos_x, pos_z)    

plt.figure(3)  
plt.plot(poincare_y, poincare_ydot, '.')   

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#ax.plot([-MU], [0.0], [0.0], 'ro')
ax.plot([1-MU],[0.0], [0.0], 'bo')
ax.plot(pos_x, pos_y, pos_z, 'g')

plt.show()    


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


    
    