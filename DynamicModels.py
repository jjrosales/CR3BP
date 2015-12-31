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

class CRTBP_DynSys(object):

    def __init__(self, mu):

        self.dim = 6
        self.mu  = mu
        
        # Defined for convenience
        self.mu_2 = mu
        self.mu_1 = 1.0 - mu
        
        self.is_valid        = False
        self.cond__init_flag = False
        self.t0_flag         = False
        self.tf_flag         = False
        
        self.exec_ok         = False
        
        self.intial_condition = np.zeros(self.dim, dtype=np.double)
        self.state_vector     = np.zeros(self.dim, dtype=np.double)
        self.f_eval           = np.zeros(self.dim, dtype=np.double)
      
        self.JC = 0.0        
        
        self.t0 = 0.0
        self.tf = 0.0
        
        # numerical intergrator        
        self.__odeint = ode(self.__f).set_integrator('dopri5')


        #Public attibute
        self.L = np.zeros((5,3), dtype=np.double)
        
        # computes libation points and initalizes self.L
        self.__libration_points()

    # This method sets the value of the state vector.
    def set_initial_condition(self, cond_ini):
        self.intial_condition = cond_ini
        self.state_vector     = cond_ini
        self.cond__init_flag =  True

    # This method sets the intial time.
    def set_t0(self, t0):
        self.t0 = t0
        self.t0_flag   = True
        
    # This method sets the final time.
    def set_tf(self, tf):
        self.tf = tf
        self.tf_flag   = True        

    # Integrates the dynamic model from t0 to tf at dt steps
    def go(self):
        if self.is_valid_():

           self.__odeint.set_initial_value(self.intial_condition, self.t0)
           self.state_vector = self.__odeint.integrate(self.tf)

           self.exec_ok = self.__odeint.successful()

    # Returns the updated state vector
    def get_updated_state_vector(self):
        if not self.exec_ok:
            self.__err = -1
        return self.state_vector

    # Returns the updated time
    def get_updated_time(self):
        if not self.exec_ok:
            self.__err = -1
        return self.tf


    # Returns the Jacobi Constant
    def get_Jacobi_Constant(self):
        if self.cond__init_flag:
            
            x_minus_mu1 = self.state_vector[0] - self.mu_1
            x_plus_mu2  = self.state_vector[0] + self.mu_2 
            
            y2 = self.state_vector[1]*self.state_vector[1]
            z2 = self.state_vector[2]*self.state_vector[2]            

            x_minus_mu1 = self.state_vector[0] - self.mu_1
            x_plus_mu2  = self.state_vector[0] + self.mu_2  
            
            aux  = y2 + z2           
            
            r1 = aux + x_plus_mu2*x_plus_mu2
            r1 = np.sqrt(r1)
            
            r2 = aux + x_minus_mu1*x_minus_mu1
            r2 = np.sqrt(r2)
    
            U =     self.mu_1*r1*r1 
            U = U + self.mu_2*r2*r2
            U = -0.5*U
            U = U - self.mu_1/r1
            U = U - self.mu_2/r2
    
            self.__JC =           - (self.state_vector[3]*self.state_vector[3]) 
            self.__JC = self.__JC - (self.state_vector[4]*self.state_vector[4])             
            self.__JC = self.__JC - (self.state_vector[5]*self.state_vector[5])
            self.__JC = self.__JC - 2.0*U

        return self.__JC
        
    def get_jacobian(self):

            
        J_CRTBP = np.zeros((self.dim, self.dim), dtype=np.double)
        
        x_minus_mu1 = self.state_vector[0] - self.mu_1
        x_plus_mu2  = self.state_vector[0] + self.mu_2         
        
        x_12 = x_plus_mu2  * x_plus_mu2             
        x_22 = x_minus_mu1 * x_minus_mu1

        y2   =  self.state_vector[1] * self.state_vector[1]
        z2   =  self.state_vector[2] * self.state_vector[2]

        aux  = y2 + z2
        
        r2   = aux + x_22
        r23  = r2*r2*r2
        r2_3 = self.mu_2/np.sqrt(r23)
        r2_5 = r2_3/r2
        
        r1   = aux + x_12
        r13  = r1*r1*r1
        r1_3 = self.mu_1/np.sqrt(r1)
        r1_5 = r1_3/r1


        df4_aux = x_minus_mu1*r1_5 + x_plus_mu2*r2_5 
        r1_3_plus_r2_3 = r2_3 + r1_3
        r1_5_plus_r2_5 = r2_5 + r1_5
       
        J_CRTBP[0,3] = 1.0
        J_CRTBP[1,4] = 1.0
        J_CRTBP[2,5] = 1.0
        
        J_CRTBP[3,4] = 2.0
        J_CRTBP[4,3] = -2.0
        
        J_CRTBP[3,0] = 1 - r1_3_plus_r2_3          
        J_CRTBP[3,0] = J_CRTBP[3,0] + 3.0*(r2_5*x_22 + r1_5*x_12) 
        
        J_CRTBP[3,1] = 3.0*self.state_vector[1] * df4_aux
        J_CRTBP[3,2] = 3.0*self.state_vector[2] * df4_aux
        
        
        J_CRTBP[4,0] = J_CRTBP[3,1]
        
        J_CRTBP[4,1] = 1 - r1_3_plus_r2_3           
        J_CRTBP[4,1] = J_CRTBP[4,1] + 3.0*y2*r1_5_plus_r2_5 
        
        J_CRTBP[4,2] = 3.0*self.state_vector[1]*self.state_vector[2]*r1_5_plus_r2_5

        
        J_CRTBP[5,0] = J_CRTBP[3,2]
        J_CRTBP[5,1] = J_CRTBP[4,2]
        J_CRTBP[5,2] = r1_3_plus_r2_3 + 3.0*z2*r1_5_plus_r2_5
 
        return J_CRTBP            
        
        
    def get_exec_flag(self):
        return self.exec_ok

    def get_f_eval(self):
        return self.f_eval
        
    def get_dim(self):
        return self.dim

    # Evaluates the field
    def __f (self, time, state_vector):
     
        state_update = np.zeros(self.dim, dtype=np.double)
        
        x_minus_mu1 = state_vector[0] - self.mu_1
        x_plus_mu2  = state_vector[0] + self.mu_2         
        
        x_12 = x_plus_mu2  * x_plus_mu2             
        x_22 = x_minus_mu1 * x_minus_mu1
        
        y2   =  state_vector[1] * state_vector[1]
        z2   =  state_vector[2] * state_vector[2]

        aux  = y2 + z2
        
        r2   = aux + x_22
        r2   = r2*r2*r2
        r2_3 = self.mu_2/np.sqrt(r2)
        
        r1   = aux + x_12
        r1   = r1*r1*r1
        r1_3 = self.mu_1/np.sqrt(r1)

        state_update[0] = state_vector[3] 
        
        state_update[1] = state_vector[4]

        state_update[2] = state_vector[5]
        
        state_update[3] =                   2.0*state_vector[4] + state_vector[0]
        state_update[3] = state_update[3] - x_plus_mu2*r1_3
        state_update[3] = state_update[3] - x_minus_mu1*r2_3 
        
        state_update[4] =                 - 2.0*state_vector[3] + state_vector[1]
        state_update[4] = state_update[4] - state_vector[1]*r2_3
        state_update[4] = state_update[4] - state_vector[1]*r1_3 
        
        state_update[5] =                 - state_vector[2]*r2_3
        state_update[5] = state_update[5] - state_vector[2]*r1_3

        self.f_eval = state_update
 
        return self.f_eval
        
    # Evaluates the field
    def __libration_points (self):
      
       tol   = 1e-12
       maxit = 25
       norm = 999.0
       i    = 0
      
       # Intial Approximations      
      
       L1 = (self.mu/3.)**(1/3.)
       L2 = L1
       L3 = 1.0 - (7./12.)*self.mu

       # Computes L1  
       while norm > tol:
           
           x =   L1 - (3.0 - self.mu)
           x = x*L1 + (3.0 - 2.0*self.mu)
           x = x*L1 - self.mu
           x = x*L1 + 2.0*self.mu
           x = x*L1 - self.mu
           
           dx = 5.0*L1 - 4.0*(3.0 - self.mu) 
           dx =  dx*L1 + 3.0*(3.0 - 2.0*self.mu)
           dx =  dx*L1 - 2.0*self.mu
           dx =  dx*L1 + 2.0*self.mu
           
           norm = abs(x)

           if norm > tol:
               dx = x/dx
               L1 = L1 - dx
           elif norm <= tol:
               norm = 999.0
               i    = 0
               L1 = self.mu_1 - L1 
               break
           elif i>maxit:
               print '*WARNING* max. iterations reached to compute fixed points.'
               
           i = i + 1

       # Computes L2
       while norm > tol:
           
           x =   L2 + (3.0 - self.mu)
           x = x*L2 + (3.0 - 2.0*self.mu)
           x = x*L2 - self.mu
           x = x*L2 - 2.0*self.mu
           x = x*L2 - self.mu
           
           dx = 5.0*L2 + 4.0*(3.0 - self.mu) 
           dx =  dx*L2 + 3.0*(3.0 - 2.0*self.mu)
           dx =  dx*L2 - 2.0*self.mu
           dx =  dx*L2 - 2.0*self.mu
           
           
           norm = abs(x)

           if norm > tol:
               dx = x/dx
               L2 = L2 - dx
           elif norm <= tol:
               norm = 999.0
               i    = 0
               L2 = self.mu_1 + L2 
               break
           elif i>maxit:
               print '*WARNING* max. iterations reached to compute fixed points.'
               
           i = i + 1

       # Computes L3
       while norm > tol:
           x =   L3 + (2.0 + self.mu)
           x = x*L3 + (1.0 + 2.0*self.mu)
           x = x*L3 - self.mu_1
           x = x*L3 - 2.0*self.mu_1
           x = x*L3 - self.mu_1 

           dx = 5.0*L3 + 4.0*(2.0 + self.mu)
           dx =  dx*L3 + 3.0*(1.0 + 2.0*self.mu)
           dx =  dx*L3 - 2.0*self.mu_1
           dx =  dx*L3 - 2.0*self.mu_1
                    
           norm = abs(x)

           if norm > tol:
               dx = x/dx
               L3 = L3 - dx
           elif norm <= tol:
               norm = 999.0
               i    = 0
               L3 = -(self.mu_2 + L3) 
               break
           elif i>maxit:
               print '*WARNING* max. iterations reached to compute fixed points.'
               
           i = i + 1

       # Computes L4 and L5
       self.L[0,0] = L1
       self.L[1,0] = L2       
       self.L[2,0] = L3
       self.L[3,0] = 0.5 - self.mu
       self.L[4,0] = self.L[3,0] 
       self.L[3,1] = 0.5*np.sqrt(3.0)
       self.L[4,1] = -self.L[3,1] 
       
       return self.L
       
    
    # Returns 'True' if the class is ready to process the inputs. 'False' otherwise.
    def __is_valid_(self):
        if (self.cond__init_flag and
            self.t0_flag and
            self.tf_flag):
            self.is_valid = True
        else:
            self.is_valid = False
        return self.is_valid    
    
        
class CRTBP_variational_DynSys(CRTBP_DynSys):

    def __init__(self, mu):
        
        CRTBP_DynSys.__init__(self, mu)

        self.dim_var = self.dim * (1 + self.dim) 
        
        self.is_valid          = False
        self.cond__init_flag   = False
        self.variationals_flag = False
        self.t0_flag           = False
        self.tf_flag           = False
        
        self.exec_ok           = False
        
        self.intial_condition = np.zeros(self.dim, dtype=np.double)
        self.state_vector     = np.zeros(self.dim, dtype=np.double)
        
        self.variationals_ini = np.eye(self.dim, dtype=np.double)
        self.variationals     = np.eye(self.dim, dtype=np.double)


        self.extended_ini_sv  = np.zeros(self.dim_var, dtype=np.double)
        self.extended_sv      = np.zeros(self.dim_var, dtype=np.double)

        self.f_eval           = np.zeros(self.dim, dtype=np.double) # evaluation of the field f
        self.var_eval         = np.eye(self.dim*self.dim, dtype=np.double)
        self.Df_eval          = np.zeros(self.dim, dtype=np.double) # evaluation of the Jacobian Df

        # numerical intergrator        
        self.__odeint = ode(self.__f).set_integrator('dopri5')
    

    # This method sets the variationals
    def set_variationals(self, var_A):
       
        self.variationals_ini = var_A
        self.variationals     = var_A
   
        self.variationals_flag =  True
        
    def get_variational_eval(self):
        return self.var_eval
    
    def go(self):
        if self.__is_valid_():

            # constructs the extended initial state vector based upon f and the variational matrix
            self.extended_ini_sv[0:self.dim] = self.intial_condition
    
            for i in range(0,self.dim):
                self.extended_ini_sv[(i+1)*self.dim:(i+2)*self.dim] = self.variationals_ini[i, 0:self.dim]
                
#            print self.extended_ini_sv

            self.__odeint.set_initial_value(self.extended_ini_sv, self.t0)
#            print self.__odeint.integrate(self.tf)
            self.extended_sv = self.__odeint.integrate(self.tf)
            
            self.state_vector = self.extended_sv[0:self.dim]
            
            

            #            self.state_vector[0]=self.state_vector[0]+1e-5


            self.exec_ok = self.__odeint.successful()   
        else:
            
            print 'Error integration ODE. Check inputs.'
           
#        print self.dim, self.dim_var
#        print self.L
#        
#        print self.t0, self.intial_condition
        

    
    def __f(self, time, extended_state_vector):
    
        self.f_eval = CRTBP_DynSys._CRTBP_DynSys__f(self, time, extended_state_vector[0: self.dim])
        
        self.extended_sv[0:self.dim] = self.f_eval
        self.extended_sv[self.dim: self.dim_var] = extended_state_vector[self.dim: self.dim_var]
        
#        print self.extended_sv 
#        self.Df_eval = self.get_jacobian()
#        
#        self.var_eval = np.dot(self.Df_eval, self.variationals_ini)
#        
#        # constructs the extended state vector based upon f and the variational matrix
#        self.extended_sv[0:self.dim] = self.f_eval 
#
#        for i in range(0,self.dim):
#            self.extended_sv[(i+1)*self.dim:(i+2)*self.dim] = self.var_eval[i, 0:self.dim]
        
        return self.extended_sv
        
    # Returns 'True' if the class is ready to process the inputs. 'False' otherwise.
    def __is_valid_(self):
        if (self.cond__init_flag and
            self.t0_flag and
            self.tf_flag and
            self.variationals_flag):
            self.is_valid = True
        else:
            self.is_valid = False
        return self.is_valid    
            
        
        