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

   
class planar_CRTBP_DynSys:

    def __init__(self, mu):

        self.__dim = 4
        self.__mu  = mu
        
        self.__is_valid        = False
        self.__cond__init_flag = False
        self.__t0_flag         = False
        self.__tf_flag         = False
        
        self.__exec_ok         = False
        
        self.__intial_condition = np.zeros(self.__dim, dtype=np.double)
        self.__state_vector     = np.zeros(self.__dim, dtype=np.double)
        self.__f_eval           = np.zeros(self.__dim, dtype=np.double)
        
        self.__JC = 0.0        
        
        self.__t0 = 0.0
        self.__tf = 0.0
        
        # numerical intergrator        
        self.__odeint = ode(self.__f).set_integrator('dopri5')

    # This method sets the value of the state vector.
    def set_initial_condition(self, cond_ini):
        self.__intial_condition = cond_ini
        self.__cond__init_flag =  True

    # This method sets the intial time.
    def set_t0(self, t0):
        self.__t0 = t0
        self.__t0_flag   = True
        
    # This method sets the final time.
    def set_tf(self, tf):
        self.__tf = tf
        self.__tf_flag   = True        

    # Integrates the dynamic model from t0 to tf at dt steps
    def go(self):
        if self.__is_valid_():

           self.__odeint.set_initial_value(self.__intial_condition, self.__t0)
           self.__state_vector = self.__odeint.integrate(self.__tf)

           self.__exec_ok = self.__odeint.successful()

    # Returns the updated state vector
    def get_updated_state_vector(self):
        if not self.__exec_ok:
            self.__err = -1
        return self.__state_vector

    # Returns the updated time
    def get_updated_time(self):
        if not self.__exec_ok:
            self.__err = -1
        return self.__tf


    # Returns the transition matrix    
    def get_Jacobi_Constant(self):
        if self.__cond__init_flag:
            mu1 = 1 - self.__mu
            mu2 = self.__mu

            r1 = self.__state_vector[1]*self.__state_vector[1]
            r1 = r1 + (self.__state_vector[0]+mu2)*(self.__state_vector[0]+mu2)
            r1 = np.sqrt(r1)
            
            r2 = self.__state_vector[1]*self.__state_vector[1]
            r2 = r2 + (self.__state_vector[0]-mu1)*(self.__state_vector[0]-mu1)
            r2 = np.sqrt(r2)
    
            U =     mu1*r1*r1 
            U = U + mu2*r2*r2
            U = -0.5*U
            U = U - mu1/r1
            U = U - mu2/r2
    
            self.__JC =           - (self.__intial_condition[2]*self.__intial_condition[2]) 
            self.__JC = self.__JC - (self.__intial_condition[3]*self.__intial_condition[3])
            self.__JC = self.__JC - 2.0*U

        return self.__JC
        
    def get_jacobian(self):
        if self.__exec_ok:
            
            J_CRTBP = np.zeros((self.__dim, self.__dim), dtype=np.double)
            
            one_minus_mu = 1.0 - self.__mu
            
            r2 = self.__state_vector[1]*self.__state_vector[1]
            r2 = r2 + (self.__state_vector[0]-one_minus_mu)*(self.__state_vector[0]-one_minus_mu)
            r2 = np.sqrt(r2)
            
            r1 = self.__state_vector[1]*self.__state_vector[1]
            r1 = r1 + (self.__state_vector[0]+self.__mu)*(self.__state_vector[0]+self.__mu)
            r1 = np.sqrt(r1)
            
            inv_r1_3 = 1.0/(r1*r1*r1) 
            inv_r1_5 = inv_r1_3/(r1*r1)
            
            inv_r2_3 = 1.0/(r2*r2*r2) 
            inv_r2_5 = inv_r2_3/(r2*r2)
            
            
            J_CRTBP[0,2] = 1.0
            J_CRTBP[1,3] = 1.0
            
            J_CRTBP[2,3] = 2.0
            J_CRTBP[3,2] = -2.0
            
            J_CRTBP[2,0] = one_minus_mu*(-inv_r1_3 + 3.0*(self.__state_vector[0]+self.__mu)*(self.__state_vector[0]+self.__mu)*inv_r1_5)
            J_CRTBP[2,0] = J_CRTBP[2,0] + self.__mu*(-inv_r2_3 + 3.0*(self.__state_vector[0]-one_minus_mu)*(self.__state_vector[0]-one_minus_mu)*inv_r2_5)
            J_CRTBP[2,0] = J_CRTBP[2,0] + 1.0    
            
            J_CRTBP[2,1] = 3.0*one_minus_mu*(self.__state_vector[0]+self.__mu)*self.__state_vector[1]*inv_r2_5
            J_CRTBP[2,1] = J_CRTBP[2,1] + 3.0*self.__mu*(self.__state_vector[0]-one_minus_mu)*self.__state_vector[1]*inv_r1_5
            
            J_CRTBP[3,0] = J_CRTBP[2,1]
            
            J_CRTBP[3,1] = one_minus_mu*(-inv_r1_3 + 3.0*self.__state_vector[1]*self.__state_vector[1]*inv_r1_5)
            J_CRTBP[3,1] = J_CRTBP[3,1] + self.__mu*(-inv_r2_3 + 3.0*self.__state_vector[1]*self.__state_vector[1]*inv_r2_5)
            J_CRTBP[3,1] = J_CRTBP[3,1] + 1.0       
 
        return J_CRTBP            
        
        
    def get_exec_flag(self):
        return self.__exec_ok

    def get_f_eval(self):
        return self.__f_eval
        
    def get_dim(self):
        return self.__dim

    # Evaluates the field
    def __f (self, time, state_vector):
        
        one_minus_mu = 1.0 - self.__mu
        
        state_update = np.zeros(self.__dim, dtype=np.double)
        
        r2 = state_vector[1]*state_vector[1]
        r2 = r2 + (state_vector[0]-one_minus_mu)*(state_vector[0]-one_minus_mu)
        r2 = r2*r2*r2
        r2 = np.sqrt(r2)
        
        r1 = state_vector[1]*state_vector[1]
        r1 = r1 + (state_vector[0]+self.__mu)*(state_vector[0]+self.__mu)
        r1 = r1*r1*r1
        r1 = np.sqrt(r1)
     
        state_update[0] = state_vector[2] 
        
        state_update[1] = state_vector[3]
        
        state_update[2] = 2.0*state_vector[3] + state_vector[0]
        state_update[2] = state_update[2] - self.__mu*(state_vector[0]-one_minus_mu)/r2
        state_update[2] = state_update[2] - one_minus_mu*(state_vector[0]+self.__mu)/r1 
        
        state_update[3] =-2.0*state_vector[2] + state_vector[1]
        state_update[3] = state_update[3] - self.__mu*(state_vector[1])/r2
        state_update[3] = state_update[3] - one_minus_mu*state_vector[1]/r1 

        self.__f_eval = state_update
 
        return self.__f_eval

    
    # Returns 'True' if the class is ready to process the inputs. 'False' otherwise.
    def __is_valid_(self):
        if (self.__cond__init_flag and
            self.__t0_flag and
            self.__tf_flag):
            self.__is_valid = True
        else:
            self.__is_valid = False
        return self.__is_valid    
    
#
# 3D CRTBP
#        

class CRTBP_DynSys:

    def __init__(self, mu):

        self.__dim = 6
        self.__mu  = mu
        
        self.__is_valid        = False
        self.__cond__init_flag = False
        self.__t0_flag         = False
        self.__tf_flag         = False
        
        self.__exec_ok         = False
        
        self.__intial_condition = np.zeros(self.__dim, dtype=np.double)
        self.__state_vector     = np.zeros(self.__dim, dtype=np.double)
        self.__f_eval           = np.zeros(self.__dim, dtype=np.double)
        
        self.__JC = 0.0        
        
        self.__t0 = 0.0
        self.__tf = 0.0
        
        # numerical intergrator        
        self.__odeint = ode(self.__f).set_integrator('dopri5')

    # This method sets the value of the state vector.
    def set_initial_condition(self, cond_ini):
        self.__intial_condition = cond_ini
        self.__cond__init_flag =  True

    # This method sets the intial time.
    def set_t0(self, t0):
        self.__t0 = t0
        self.__t0_flag   = True
        
    # This method sets the final time.
    def set_tf(self, tf):
        self.__tf = tf
        self.__tf_flag   = True        

    # Integrates the dynamic model from t0 to tf at dt steps
    def go(self):
        if self.__is_valid_():

           self.__odeint.set_initial_value(self.__intial_condition, self.__t0)
           self.__state_vector = self.__odeint.integrate(self.__tf)

           self.__exec_ok = self.__odeint.successful()

    # Returns the updated state vector
    def get_updated_state_vector(self):
        if not self.__exec_ok:
            self.__err = -1
        return self.__state_vector

    # Returns the updated time
    def get_updated_time(self):
        if not self.__exec_ok:
            self.__err = -1
        return self.__tf


    # Returns the transition matrix    
    def get_Jacobi_Constant(self):
        if self.__cond__init_flag:
            mu1 = 1 - self.__mu
            mu2 = self.__mu

            r1 =      self.__state_vector[1]*self.__state_vector[1]
            r1 = r1 + self.__state_vector[2]*self.__state_vector[2]
            r1 = r1 + (self.__state_vector[0]+mu2)*(self.__state_vector[0]+mu2)
            r1 = np.sqrt(r1)
            
            r2 =      self.__state_vector[1]*self.__state_vector[1]
            r2 = r2 + self.__state_vector[2]*self.__state_vector[2]            
            r2 = r2 + (self.__state_vector[0]-mu1)*(self.__state_vector[0]-mu1)
            r2 = np.sqrt(r2)
    
            U =     mu1*r1*r1 
            U = U + mu2*r2*r2
            U = -0.5*U
            U = U - mu1/r1
            U = U - mu2/r2
    
            self.__JC =           - (self.__intial_condition[2]*self.__intial_condition[2]) 
            self.__JC = self.__JC - (self.__intial_condition[3]*self.__intial_condition[3])             
            self.__JC = self.__JC - (self.__intial_condition[4]*self.__intial_condition[4])
            self.__JC = self.__JC - 2.0*U

        return self.__JC
        
    def get_jacobian(self):
        if self.__exec_ok:
            
            J_CRTBP = np.zeros((self.__dim, self.__dim), dtype=np.double)
            
            one_minus_mu = 1.0 - self.__mu
            
            r2 = self.__state_vector[1]*self.__state_vector[1]
            r2 = r2 + (self.__state_vector[0]-one_minus_mu)*(self.__state_vector[0]-one_minus_mu)
            r2 = np.sqrt(r2)
            
            r1 = self.__state_vector[1]*self.__state_vector[1]
            r1 = r1 + (self.__state_vector[0]+self.__mu)*(self.__state_vector[0]+self.__mu)
            r1 = np.sqrt(r1)
            
            inv_r1_3 = 1.0/(r1*r1*r1) 
            inv_r1_5 = inv_r1_3/(r1*r1)
            
            inv_r2_3 = 1.0/(r2*r2*r2) 
            inv_r2_5 = inv_r2_3/(r2*r2)
            
            
            J_CRTBP[0,2] = 1.0
            J_CRTBP[1,3] = 1.0
            J_CRTBP[2,4] = 1.0
            
            J_CRTBP[3,4] = 2.0
            J_CRTBP[4,3] = -2.0
            
            J_CRTBP[3,0] = one_minus_mu*(-inv_r1_3 + 3.0*(self.__state_vector[0]+self.__mu)*(self.__state_vector[0]+self.__mu)*inv_r1_5)
            J_CRTBP[3,0] = J_CRTBP[2,0] + self.__mu*(-inv_r2_3 + 3.0*(self.__state_vector[0]-one_minus_mu)*(self.__state_vector[0]-one_minus_mu)*inv_r2_5)
            J_CRTBP[3,0] = J_CRTBP[2,0] + 1.0    
            
            J_CRTBP[3,1] = 3.0*one_minus_mu*(self.__state_vector[0]+self.__mu)*self.__state_vector[1]*inv_r2_5
            J_CRTBP[3,1] = J_CRTBP[2,1] + 3.0*self.__mu*(self.__state_vector[0]-one_minus_mu)*self.__state_vector[1]*inv_r1_5
            
            J_CRTBP[4,0] = J_CRTBP[2,1]
            
            J_CRTBP[4,1] = one_minus_mu*(-inv_r1_3 + 3.0*self.__state_vector[1]*self.__state_vector[1]*inv_r1_5)
            J_CRTBP[4,1] = J_CRTBP[3,1] + self.__mu*(-inv_r2_3 + 3.0*self.__state_vector[1]*self.__state_vector[1]*inv_r2_5)
            J_CRTBP[4,1] = J_CRTBP[3,1] + 1.0       
 
        return J_CRTBP            
        
        
    def get_exec_flag(self):
        return self.__exec_ok

    def get_f_eval(self):
        return self.__f_eval
        
    def get_dim(self):
        return self.__dim

    # Evaluates the field
    def __f (self, time, state_vector):
        
        one_minus_mu = 1.0 - self.__mu
        
        state_update = np.zeros(self.__dim, dtype=np.double)
        
        r2 = state_vector[1]*state_vector[1]
        r2 = r2 + (state_vector[0]-one_minus_mu)*(state_vector[0]-one_minus_mu)
        r2 = r2*r2*r2
        r2 = np.sqrt(r2)
        
        r1 = state_vector[1]*state_vector[1]
        r1 = r1 + (state_vector[0]+self.__mu)*(state_vector[0]+self.__mu)
        r1 = r1*r1*r1
        r1 = np.sqrt(r1)
     
        state_update[0] = state_vector[3] 
        
        state_update[1] = state_vector[4]

        state_update[2] = state_vector[5]
        
        state_update[3] =                   2.0*state_vector[4] + state_vector[0]
        state_update[3] = state_update[3] - self.__mu*(state_vector[0]-one_minus_mu)/r2
        state_update[3] = state_update[3] - one_minus_mu*(state_vector[0]+self.__mu)/r1 
        
        state_update[4] =                 - 2.0*state_vector[3] + state_vector[1]
        state_update[4] = state_update[4] - self.__mu*(state_vector[1])/r2
        state_update[4] = state_update[4] - one_minus_mu*state_vector[1]/r1 
        
        state_update[5] =                 - self.__mu*state_vector[2]/r2
        state_update[5] = state_update[5] - one_minus_mu*state_vector[2]/r1

        self.__f_eval = state_update
 
        return self.__f_eval

    
    # Returns 'True' if the class is ready to process the inputs. 'False' otherwise.
    def __is_valid_(self):
        if (self.__cond__init_flag and
            self.__t0_flag and
            self.__tf_flag):
            self.__is_valid = True
        else:
            self.__is_valid = False
        return self.__is_valid    
    
        

    
    
    
    