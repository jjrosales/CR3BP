__author__     = "Jose Rosales"
__copyright__  = "TBD"
__credits__    = "TBD"
__license__    = "TBD"
__maintainer__ = "Jose Rosales"
__email__      = "pepe.rosales@gmail.com"
__status__     = "Prototype"

import numpy as np


class halo_orbit:

    def __init__(self, CRTBP_DynSys, libration_point):
        
        
        self._dim       = CRTBP_DynSys._dim 
        self._ini_state = np.zeros(self._dim, dtype=np.double) 
        
        self._model = CRTBP_DynSys
        
        self._L_i  = libration_point - 1
        
        self._amplitude = 0.0
        
        self._mu = self._model._mu
        
        self._L_i_x_coord =  CRTBP_DynSys.get_libration_points()[self._L_i][0]
        
        self._index = 0
        
        self._tol_correction = 1e-10

        self._tol_section = 1e-12

        self._dt  = 1e-3
        
        self._time  = 0.0  
        
        self._period = 0.0        
        
        self._max_time = 6.28

        self._all_ini_cond = []          
        
    def set_amplitude(self, amplitude):
        
        self._amplitude = amplitude
        
    def go(self):
        
#        state_aux = np.zeros(6, dtype=np.double)
#
#        state_aux = self.continuation_step( [ 0.82356264 , 0.   ,       1e-2 ,      0.  ,        0.1250304 ,  0.        ])
#
#        self._ini_state[0:6] = state_aux
        
        state_aux1 = np.zeros(6, dtype=np.double)
        state_aux2 = np.zeros(6, dtype=np.double)

        aux = np.zeros(6, dtype=np.double)

        state_aux1 = self.continuation_step([0.82356264, 0., 1e-3, 0., 0.1250304, 0. ])      
        self._all_ini_cond.append([self._period, state_aux1])  
        
        state_aux2 = self.continuation_step([0.82356264, 0., 2e-3, 0., 0.1250304, 0.])
        self._all_ini_cond.append([self._period, state_aux2]) 

#        state_aux1 = self.continuation_step([1.12034502, 0., -5e-4, 0., 0.17606472, 0.])      
#        self._all_ini_cond.append([self._period, state_aux1])  
#        
#        state_aux2 = self.continuation_step([1.12034502, 0., -1e-3, 0., 0.17606472, 0.])
#        self._all_ini_cond.append([self._period, state_aux2]) 

        aux        = 2.0*state_aux2 - state_aux1
        
        for i in range(0,300):
                     
            state_aux1 = state_aux2
            state_aux2 = self.continuation_step(aux)
            aux        = 2.0*state_aux2 - state_aux1

            self._all_ini_cond.append([self._period, state_aux2])  
            
            print i, self._period, self._model.get_Jacobi_Constant(), state_aux2, -(state_aux2[0]-self._L_i_x_coord )


        self._ini_state[0:6] = state_aux2        
        
    # this method computes the initial state of a lyapunov orbit with a given
    # "small" amplitude. It is used as step in the numerical continuation to find
    # Lyapynov planar periodic orbits with "arbitrary" amplitudes
    def continuation_step(self, init_state_vec):
        
        state_vector = np.zeros(self._dim, dtype=np.double)
        ini_state    = np.zeros(self._dim, dtype=np.double)   
        
        ini_state [0:6] = init_state_vec
        ini_state [6]   = 1.0
        ini_state [13]  = 1.0
        ini_state [20]  = 1.0
        ini_state [27]  = 1.0
        ini_state [34]  = 1.0
        ini_state [41]  = 1.0
        
        self._index = 0
        
        delta_vx    = 1.0
        
        while abs(delta_vx) > self._tol_correction:
            
            self._index = self._index + 1
        
            continue_flag = True
        
            p_coord_eval_old = 0.0
            state_vector = ini_state 
            self._time   = 0.0  
        
            while abs(self._time) < self._max_time and continue_flag:
            
                # Integrate the equations of motion and the variational equations
        
                self._model.set_initial_condition(state_vector)
                self._model.set_t0(self._time)
                self._model.set_tf(self._time+self._dt)
        
                self._model.go()
                
                state_vector = self._model.get_updated_state_vector()
                time         = self._model.get_updated_time()
                
                var     = self._model.get_updated_var()
                
                self._time = self._time + self._dt
                   
                # Poincare Map computation. The section is y = 0
                p_coord_eval = state_vector[1]
                     
                if (p_coord_eval_old*p_coord_eval<0.0):   
            
                    sv_aux = state_vector
                    t_aux  = time
                    
                    delta = 0.0
            
                    while abs(p_coord_eval)>self._tol_section:
                        delta = -p_coord_eval/self._model.get_f_eval()[1]
                        self._model.set_initial_condition(sv_aux)
                        self._model.set_t0(t_aux)
                        self._model.set_tf(t_aux+delta)
            
                        self._model.go()
                        
                        sv_aux = self._model.get_updated_state_vector()
                        t_aux  = self._model.get_updated_time()
            
                        p_coord_eval = sv_aux[1]
                    
                
#                    print   t_aux,  sv_aux[0:6]
                    # stores period
                    self._period = 2.0*t_aux
                    
                    delta_vx = sv_aux[3]
                  
                    # computes the correction to be applied to the
                    # z and vy components -- we assume x constant, vel_x = vel_z = 0
                    vx_dot  = self._model.get_f_eval()[3]
                    vz_dot  = self._model.get_f_eval()[5]
                    var     = self._model.get_updated_var()
                    
                    aux_coeff_1 = vx_dot/sv_aux[4]
                    aux_coeff_2 = vz_dot/sv_aux[4]
                    
                    a11 = var[1,0]*aux_coeff_1
                    a11 = var[3,0] - a11
                    
                    a12 = var[1,4]*aux_coeff_1
                    a12 = var[3,4] - a12

                    a21 = var[1,0]*aux_coeff_2
                    a21 = var[5,0] - a21

                    a22 = var[1,4]*aux_coeff_2
                    a22 = var[5,4] - a22
                    
                    det = a11*a22 - a12*a21
                    
                    delta_x = sv_aux[3]*a22-sv_aux[5]*a12 
                    delta_x = delta_x/det
                    
                    delta_vz = sv_aux[5]*a11-sv_aux[3]*a21
                    delta_vz = delta_vz/det                    
                    
                    ini_state[0] = ini_state[0] - delta_x
                    ini_state[4] = ini_state[4] - delta_vz
                    
#                    print  delta_x,  delta_vz
                    
                    continue_flag = False
               
                p_coord_eval_old = state_vector[1]  

        if abs(self._time) >= self._max_time:
            print "** ERROR: orbit did not cross x-axis **"
            exit 
        return ini_state[0:6]
                
    def get_state_ini(self):
        # only returns positions and velocity -- we don't care about the variationals
        return self._ini_state[0:6]
        
    def get_period(self):
        return self._period 

    def get_all_ini_cond(self):
        return self._all_ini_cond
        
    def halo_seed(self, amplitude):
        
        seed_state_vector = np.zeros(6, dtype=np.double)
    
        mu1 = 1.0 - self._mu   
             
        L_plus_mu = self._L_i_x_coord + self._mu
        mu_bar = L_plus_mu - 1.0
        mu_bar = self._mu / abs(mu_bar*mu_bar*mu_bar)
        mu_bar = mu_bar + (1.0 - self._mu) * 1.0/abs(L_plus_mu*L_plus_mu*L_plus_mu)
        
        nu = (9.0*mu_bar - 8.0)*mu_bar
        nu = -np.sqrt(nu)
        nu = mu_bar - 2.0 + nu
        nu = 0.5*nu
        
        tau = nu*nu + 2.0*mu_bar + 1.0
        tau = -tau/(2.0*nu)
        
        A_x = amplitude
        v_y = -A_x*nu*tau
        
#        3.18407778087
       
        seed_state_vector[0] = self._L_i_x_coord - A_x
        seed_state_vector[1] = 0.0
        seed_state_vector[2] = 0.0
        seed_state_vector[3] = 0.0
        seed_state_vector[4] = v_y
        seed_state_vector[5] = mu_bar
       
        return seed_state_vector 

