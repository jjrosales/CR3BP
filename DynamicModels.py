__author__     = "Jose Rosales"
__copyright__  = "TBD"
__credits__    = "TBD"
__license__    = "TBD"
__maintainer__ = "Jose Rosales"
__email__      = "pepe.rosales@gmail.com"
__status__     = "Prototype"

__version__ = "$Revision: 1a40d4eaa00b $"

# $Source$

import numpy as np
from   Constants import *
from   scipy.integrate import ode


# 3D CRTBP
#        

class CRTBP_DynSys:

    def __init__(self, mu):

        self._dim = 42
        self._mu  = mu
        
        # Defined for convenience
        self._mu_2 = mu
        self._mu_1 = 1.0 - mu
        
        self._is_valid        = False
        self._cond__init_flag = False
        self._t0_flag         = False
        self._tf_flag         = False
        
        self._exec_ok         = False
        
        self._intial_condition = np.zeros(self._dim, dtype=np.double)
        self._state_vector     = np.zeros(self._dim, dtype=np.double)
        self._f_eval           = np.zeros(self._dim, dtype=np.double)

        self._L                = np.zeros((5,3), dtype=np.double)
        
        self._JC = 0.0        
        
        self._t0 = 0.0
        self._tf = 0.0
        
        # numerical intergrator        
        self._odeint = ode(self.__f).set_integrator('dopri5')
        
        # computes libation points and initalized self._L
        self.__libration_points()

    # This method sets the value of the state vector.
    def set_initial_condition(self, cond_ini):
        self._intial_condition = cond_ini
        self._state_vector     = cond_ini
        self._cond__init_flag =  True

    # This method sets the intial time.
    def set_t0(self, t0):
        self._t0 = t0
        self._t0_flag   = True
        
    # This method sets the final time.
    def set_tf(self, tf):
        self._tf = tf
        self._tf_flag   = True        

    # Integrates the dynamic model from t0 to tf at dt steps
    def go(self):
        if self._is_valid_():

           self._odeint.set_initial_value(self._intial_condition, self._t0)
           self._state_vector = self._odeint.integrate(self._tf)

           self._exec_ok = self._odeint.successful()

    # Returns the updated state vector
    def get_updated_state_vector(self):
        if not self._exec_ok:
            self.__err = -1
        return self._state_vector

    # Returns the updated time
    def get_updated_time(self):
        if not self._exec_ok:
            self.__err = -1
        return self._tf

    # Returns the libration points in the form of a 5x3 matrix. The first raw is L1 and so on...
    def get_libration_points(self):
        return self._L

    # Returns the Jacobi Constant
    def get_Jacobi_Constant(self):
        if self._cond__init_flag:
            
            x_minus_mu1 = self._state_vector[0] - self._mu_1
            x_plus_mu2  = self._state_vector[0] + self._mu_2 
            
            y2 = self._state_vector[1]*self._state_vector[1]
            z2 = self._state_vector[2]*self._state_vector[2]            

            x_minus_mu1 = self._state_vector[0] - self._mu_1
            x_plus_mu2  = self._state_vector[0] + self._mu_2  
            
            aux  = y2 + z2           
            
            r1 = aux + x_plus_mu2*x_plus_mu2
            r1 = np.sqrt(r1)
            
            r2 = aux + x_minus_mu1*x_minus_mu1
            r2 = np.sqrt(r2)
    
            U =     self._mu_1*r1*r1 
            U = U + self._mu_2*r2*r2
            U = -0.5*U
            U = U - self._mu_1/r1
            U = U - self._mu_2/r2
    
            self._JC =          - (self._state_vector[3]*self._state_vector[3]) 
            self._JC = self._JC - (self._state_vector[4]*self._state_vector[4])             
            self._JC = self._JC - (self._state_vector[5]*self._state_vector[5])
            self._JC = self._JC - 2.0*U

        return self._JC
        
    def get_exec_flag(self):
        return self._exec_ok

    def get_f_eval(self):
        return self._f_eval
        
    def get_dim(self):
        return self._dim

    # Evaluates the field
    def __f (self, time, state_vector):
     
        state_update = np.zeros(self._dim, dtype=np.double)
        J_CRTBP      = np.zeros((self._dim, self._dim), dtype=np.double)
 
        x_minus_mu1 = state_vector[0] - self._mu_1
        x_plus_mu2  = state_vector[0] + self._mu_2         
        
        x_12 = x_plus_mu2  * x_plus_mu2             
        x_22 = x_minus_mu1 * x_minus_mu1
        
        y2   =  state_vector[1] * state_vector[1]
        z2   =  state_vector[2] * state_vector[2]

        aux  = y2 + z2
        
        r2   = aux + x_22
        r23  = r2*r2*r2
        r2_3 = self._mu_2/np.sqrt(r23)
        r2_5 = r2_3/r2
        
        r1   = aux + x_12
        r13  = r1*r1*r1
        r1_3 = self._mu_1/np.sqrt(r13)
        r1_5 = r1_3/r1


        df4_aux = x_minus_mu1*r1_5 + x_plus_mu2*r2_5 
        r1_3_plus_r2_3 = r2_3 + r1_3
        r1_5_plus_r2_5 = r2_5 + r1_5

#---------------------
           
        J_CRTBP[0,3] = 1.0
        J_CRTBP[1,4] = 1.0
        J_CRTBP[2,5] = 1.0 * 0
        
        J_CRTBP[3,4] = 2.0
        J_CRTBP[4,3] = -2.0
        
        J_CRTBP[3,0] = 1 - r1_3_plus_r2_3          
        J_CRTBP[3,0] = J_CRTBP[3,0] + 3.0*(r2_5*x_22 + r1_5*x_12) 
        
        J_CRTBP[3,1] = 3.0*state_vector[1] * df4_aux
        J_CRTBP[3,2] = 3.0*state_vector[2] * df4_aux
                
        J_CRTBP[4,0] = J_CRTBP[3,1]
        
        J_CRTBP[4,1] = 1 - r1_3_plus_r2_3           
        J_CRTBP[4,1] = J_CRTBP[4,1] + 3.0*y2*r1_5_plus_r2_5 
        
        J_CRTBP[4,2] = 3.0*state_vector[1]*state_vector[2]*r1_5_plus_r2_5
        
        J_CRTBP[5,0] = J_CRTBP[3,2] * 0
        J_CRTBP[5,1] = J_CRTBP[4,2] * 0
        J_CRTBP[5,2] = (r1_3_plus_r2_3 + 3.0*z2*r1_5_plus_r2_5) * 0
 
#---------------------

        state_update[0]  = state_vector[3] 
        
        state_update[1]  = state_vector[4]

        state_update[2]  = state_vector[5]
        
        state_update[3]  =                   2.0*state_vector[4] + state_vector[0]
        state_update[3]  = state_update[3] - x_plus_mu2*r1_3
        state_update[3]  = state_update[3] - x_minus_mu1*r2_3 
        
        state_update[4]  =                 - 2.0*state_vector[3] + state_vector[1]
        state_update[4]  = state_update[4] - state_vector[1]*r2_3
        state_update[4]  = state_update[4] - state_vector[1]*r1_3 
        
        state_update[5]  =                 - state_vector[2]*r2_3
        state_update[5]  = state_update[5] - state_vector[2]*r1_3
        
        state_update[6]  = J_CRTBP[3,0]*state_vector[9] + J_CRTBP[4,0]*state_vector[10] + J_CRTBP[5,0]*state_vector[11]
        state_update[7]  = J_CRTBP[3,1]*state_vector[9] + J_CRTBP[4,1]*state_vector[10] + J_CRTBP[5,1]*state_vector[11]
        state_update[8]  = J_CRTBP[3,2]*state_vector[9] + J_CRTBP[4,2]*state_vector[10] + J_CRTBP[5,2]*state_vector[11]
        state_update[9]  = state_vector[6] - 2.0*state_vector[10]
        state_update[10] = state_vector[7] + 2.0*state_vector[9]
        state_update[11] = state_vector[8]

        state_update[12] = J_CRTBP[3,0]*state_vector[15] + J_CRTBP[4,0]*state_vector[16] + J_CRTBP[5,0]*state_vector[17]
        state_update[13] = J_CRTBP[3,1]*state_vector[15] + J_CRTBP[4,1]*state_vector[16] + J_CRTBP[5,1]*state_vector[17]
        state_update[14] = J_CRTBP[3,2]*state_vector[15] + J_CRTBP[4,2]*state_vector[16] + J_CRTBP[5,2]*state_vector[17]
        state_update[15] = state_vector[12] - 2.0*state_vector[16]
        state_update[16] = state_vector[13] + 2.0*state_vector[15]
        state_update[17] = state_vector[14]

        state_update[18] = J_CRTBP[3,0]*state_vector[21] + J_CRTBP[4,0]*state_vector[22] + J_CRTBP[5,0]*state_vector[23]
        state_update[19] = J_CRTBP[3,1]*state_vector[21] + J_CRTBP[4,1]*state_vector[22] + J_CRTBP[5,1]*state_vector[23]
        state_update[20] = J_CRTBP[3,2]*state_vector[21] + J_CRTBP[4,2]*state_vector[22] + J_CRTBP[5,2]*state_vector[23]
        state_update[21] = state_vector[18] - 2.0*state_vector[22]
        state_update[22] = state_vector[19] + 2.0*state_vector[21]
        state_update[23] = state_vector[20]

        state_update[24] = J_CRTBP[3,0]*state_vector[27] + J_CRTBP[4,0]*state_vector[28] + J_CRTBP[5,0]*state_vector[29]
        state_update[25] = J_CRTBP[3,1]*state_vector[27] + J_CRTBP[4,1]*state_vector[28] + J_CRTBP[5,1]*state_vector[29]
        state_update[26] = J_CRTBP[3,2]*state_vector[27] + J_CRTBP[4,2]*state_vector[28] + J_CRTBP[5,2]*state_vector[29]
        state_update[27] = state_vector[24] - 2.0*state_vector[28]
        state_update[28] = state_vector[25] + 2.0*state_vector[27]
        state_update[29] = state_vector[26]
        
        state_update[30] = J_CRTBP[3,0]*state_vector[33] + J_CRTBP[4,0]*state_vector[34] + J_CRTBP[5,0]*state_vector[35]
        state_update[31] = J_CRTBP[3,1]*state_vector[33] + J_CRTBP[4,1]*state_vector[34] + J_CRTBP[5,1]*state_vector[35]
        state_update[32] = J_CRTBP[3,2]*state_vector[33] + J_CRTBP[4,2]*state_vector[34] + J_CRTBP[5,2]*state_vector[35]
        state_update[33] = state_vector[30] - 2.0*state_vector[34]
        state_update[34] = state_vector[31] + 2.0*state_vector[33]
        state_update[35] = state_vector[32]

        state_update[36] = J_CRTBP[3,0]*state_vector[39] + J_CRTBP[4,0]*state_vector[40] + J_CRTBP[5,0]*state_vector[41]
        state_update[37] = J_CRTBP[3,1]*state_vector[39] + J_CRTBP[4,1]*state_vector[40] + J_CRTBP[5,1]*state_vector[41]
        state_update[38] = J_CRTBP[3,2]*state_vector[39] + J_CRTBP[4,2]*state_vector[40] + J_CRTBP[5,2]*state_vector[41]
        state_update[39] = state_vector[36] - 2.0*state_vector[39]
        state_update[40] = state_vector[37] + 2.0*state_vector[40]
        state_update[41] = state_vector[38]
        
        self._f_eval = state_update
 
        return self._f_eval
        
    # Evaluates the field
    def __libration_points (self):
      
       tol   = 1e-12
       maxit = 25
       norm = 999.0
       i    = 0
      
       # Intial Approximations      
      
       L1 = (self._mu/3.)**(1/3.)
       L2 = L1
       L3 = 1.0 - (7./12.)*self._mu

       # Computes L1  
       while norm > tol:
           
           x =   L1 - (3.0 - self._mu)
           x = x*L1 + (3.0 - 2.0*self._mu)
           x = x*L1 - self._mu
           x = x*L1 + 2.0*self._mu
           x = x*L1 - self._mu
           
           dx = 5.0*L1 - 4.0*(3.0 - self._mu) 
           dx =  dx*L1 + 3.0*(3.0 - 2.0*self._mu)
           dx =  dx*L1 - 2.0*self._mu
           dx =  dx*L1 + 2.0*self._mu
           
           norm = abs(x)

           if norm > tol:
               dx = x/dx
               L1 = L1 - dx
           elif norm <= tol:
               norm = 999.0
               i    = 0
               L1 = self._mu_1 - L1 
               break
           elif i>maxit:
               print '*WARNING* max. iterations reached to compute fixed points.'
               
           i = i + 1

       # Computes L2
       while norm > tol:
           
           x =   L2 + (3.0 - self._mu)
           x = x*L2 + (3.0 - 2.0*self._mu)
           x = x*L2 - self._mu
           x = x*L2 - 2.0*self._mu
           x = x*L2 - self._mu
           
           dx = 5.0*L2 + 4.0*(3.0 - self._mu) 
           dx =  dx*L2 + 3.0*(3.0 - 2.0*self._mu)
           dx =  dx*L2 - 2.0*self._mu
           dx =  dx*L2 - 2.0*self._mu
           
           
           norm = abs(x)

           if norm > tol:
               dx = x/dx
               L2 = L2 - dx
           elif norm <= tol:
               norm = 999.0
               i    = 0
               L2 = self._mu_1 + L2 
               break
           elif i>maxit:
               print '*WARNING* max. iterations reached to compute fixed points.'
               
           i = i + 1

       # Computes L3
       while norm > tol:
           x =   L3 + (2.0 + self._mu)
           x = x*L3 + (1.0 + 2.0*self._mu)
           x = x*L3 - self._mu_1
           x = x*L3 - 2.0*self._mu_1
           x = x*L3 - self._mu_1 

           dx = 5.0*L3 + 4.0*(2.0 + self._mu)
           dx =  dx*L3 + 3.0*(1.0 + 2.0*self._mu)
           dx =  dx*L3 - 2.0*self._mu_1
           dx =  dx*L3 - 2.0*self._mu_1
                    
           norm = abs(x)

           if norm > tol:
               dx = x/dx
               L3 = L3 - dx
           elif norm <= tol:
               norm = 999.0
               i    = 0
               L3 = -(self._mu_2 + L3) 
               break
           elif i>maxit:
               print '*WARNING* max. iterations reached to compute fixed points.'
               
           i = i + 1

       # Computes L4 and L5
       self._L[0,0] = L1
       self._L[1,0] = L2       
       self._L[2,0] = L3
       self._L[3,0] = 0.5 - self._mu
       self._L[4,0] = self._L[3,0] 
       self._L[3,1] = 0.5*np.sqrt(3.0)
       self._L[4,1] = -self._L[3,1] 
       
       return self._L
       
    
    # Returns 'True' if the class is ready to process the inputs. 'False' otherwise.
    def _is_valid_(self):
        if (self._cond__init_flag and
            self._t0_flag and
            self._tf_flag):
            self._is_valid = True
        else:
            self._is_valid = False
        return self._is_valid    
    
        

    
    
    
    