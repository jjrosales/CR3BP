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
from matplotlib import cm
from scipy.integrate import ode
import DynamicModels
import PoincareSections
from Lyapunov_seed import * 
import planar_lyapunov as planar_lyapunov 
import vertical_lyapunov as vertical_lyapunov 
import halo_orbit as halo_orbit
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

i = 0

L_i = DynSys.get_libration_points()[i][0]

planar_lyapunov_po   =  planar_lyapunov.planar_lyapunov(DynSys, i+1)
vertical_lyapunov_po =  vertical_lyapunov.vertical_lyapunov(DynSys, i+1)
halo_po              =  halo_orbit.halo_orbit(DynSys, i+1)

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

DynSys.set_initial_condition(ini_state_vector)
print DynSys.get_Jacobi_Constant()

#
#vertical_lyapunov_po.set_amplitude(1e-4) 
#vertical_lyapunov_po.go()
halo_po.set_amplitude(1e-4) 
halo_po.go()
#planar_lyapunov_po.set_amplitude(1e-4) 
#planar_lyapunov_po.go()

print
print 'generating plots...'
print

for i in range(301, 0, -10):
    #ini_state_vector[0:6] = vertical_lyapunov_po.get_state_ini() # [1.09132666e+00  , 0.0 , 0.0 , 0.0, -4.21422746e-01,  7.30277282e-01] 
    #ini_state_vector[0:6] = planar_lyapunov_po.get_all_ini_cond()[i][1]
    ini_state_vector[0:6] = halo_po.get_all_ini_cond()[i][1] #halo_po.get_state_ini() #
    ini_state_vector[6]   = 1.0
    ini_state_vector[13]  = 1.0
    ini_state_vector[20]  = 1.0
    ini_state_vector[27]  = 1.0
    ini_state_vector[34]  = 1.0
    ini_state_vector[41]  = 1.0
        
    T_period = halo_po.get_all_ini_cond()[i][0] #halo_po.get_period()
    #T_period = planar_lyapunov_po.get_all_ini_cond()[i][0]
    #T_period = vertical_lyapunov_po.get_period()
    
    pos_x = []
    pos_y = []
    pos_z = []
    vel_x = []
    vel_y = []
    vel_z = []
    eig_re = []
    eig_im = []
    
    state_vector = ini_state_vector
    time         = 0.0
    delta_t      = 1e-3
    
    print T_period, ini_state_vector[0:6]

    num_im = 0
   
    while abs(time) < T_period:
        
        if (time+delta_t) > T_period:
            delta_t = T_period - time
            
#        print delta_t
    
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

    eigen_values = np.linalg.eig(DynSys.get_updated_var())[0]
    
    
    for k in range(0,6):
        eig_re.append(eigen_values[k].real)
        eig_im.append(eigen_values[k].imag)        
#        print eigen_values[k].real, eigen_values[k].imag 
        
#    if abs(eigen_values[4].imag)<1e-4 and num_im<10:
#        num_im = num_im + 1
#        print T_period, ini_state_vector[0:6]

    plt.figure(0)
    plt.plot(eig_re, eig_im, 'o')    

    plt.figure(1)
    plt.xlabel('X')
    plt.ylabel('Y')  
    plt.plot(L_i, 0, 'r*')
    plt.plot(1-MU, 0, 'bo')
    plt.plot(pos_x, pos_y, 'g')
    
    plt.figure(2)
    plt.xlabel('Y')
    plt.ylabel('Z')       
    plt.plot(0, 0, 'bo')
    plt.plot(pos_y, pos_z, 'g')
    
    plt.figure(3)
    plt.xlabel('X')
    plt.ylabel('Z')      
    plt.plot(L_i, 0, 'r*')
    plt.plot(1-MU, 0, 'bo')
    plt.plot(pos_x, pos_z, 'g')
    
    fig = plt.figure(5)
    plt.hold(True)
    
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot([L_i], [0.0], [0.0], 'r*')
    ax.plot([1-MU], [0.0], [0.0], 'bo')
    ax.plot(pos_x, pos_y, pos_z, 'g')
    
    plt.show()    
    plt.axis('equal')
