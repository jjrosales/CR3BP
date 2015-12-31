__author__     = "Jose Rosales"
__copyright__  = "TBD"
__credits__    = "TBD"
__license__    = "TBD"
__maintainer__ = "Jose Rosales"
__email__      = "pepe.rosales@gmail.com"
__status__     = "Prototype"

#
# The Poincare section ia function g() that defines a hyperspace E in the state space
# The idea is that is a point x in an R^n space belongs to the hyperspace E defined by g(),
# then g(x) = 0.
#
# The classes in this file evalute the value of a point x to the function g
#

import numpy as np

class g_poincare_planar:

# The poincare section considered here is characterized as ball of center
# self.__c and radius self.__r 

    def __init__(self):

        self.__dim = 4
        
        self.__c  = np.zeros(self.__dim, dtype=np.double)
        self.__x  = np.zeros(self.__dim, dtype=np.double)
        self.__dg = np.zeros(self.__dim, dtype=np.double)

        self.__dg  = np.zeros(self.__dim, dtype=np.double)

        self.__r = 0.0
          
        self.__eval = 0.0        
        
    # this method sets the center of the neigborhood considered as the domain for the Poincare section    
    def set_center(self, center):    
        self.__c = center

    # this method sets the radius of the neigborhood considered as the domain for the Poincare section    
    def set_radius(self, radius):    
        self.__r = radius

    # This method sets the point to be evaluated.
    def set_x(self, x):
        self.__x = x
        self.__cond__init_flag =  True

    # This method sets the intial time.
    def go(self):
        self.__eval = self.__x[0] - self.__c[0]
        self.__dg   = [1.0, 0.0 ,0.0, 0.0] 

    def get_center(self):
        return self.__c  
        
    def get_radius(self):
        return self.__r   

    def get_gx(self):
        return self.__eval
        
    def get_Dg(self):
        return self.__dg
        

class g_poincare_3d:

# The poincare section considered here is characterized as ball of center
# self.__c and radius self.__r 

    def __init__(self):

        self.__dim = 6
        
        self.__c  = np.zeros(self.__dim, dtype=np.double)
        self.__x  = np.zeros(self.__dim, dtype=np.double)
        self.__dg = np.zeros(self.__dim, dtype=np.double)

        self.__dg  = np.zeros(self.__dim, dtype=np.double)

        self.__r = 0.0
          
        self.__eval = 0.0        
        
    # this method sets the center of the neigborhood considered as the domain for the Poincare section    
    def set_center(self, center):    
        self.__c = center

    # this method sets the radius of the neigborhood considered as the domain for the Poincare section    
    def set_radius(self, radius):    
        self.__r = radius

    # This method sets the point to be evaluated.
    def set_x(self, x):
        self.__x = x
        self.__cond__init_flag =  True

    # This method sets the intial time.
    def go(self):
        self.__eval =  self.__x[1] + self.__c[1]
        self.__dg   = [0.0, 1.0 ,0.0, 0.0, 0.0, 0.0] 

    def get_center(self):
        return self.__c  
        
    def get_radius(self):
        return self.__r   

    def get_gx(self):
        return self.__eval
        
    def get_Dg(self):
        return self.__dg
        
        