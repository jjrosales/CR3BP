__author__     = "Jose Rosales"
__copyright__  = "TBD"
__credits__    = "TBD"
__license__    = "TBD"
__maintainer__ = "Jose Rosales"
__email__      = "pepe.rosales@gmail.com"
__status__     = "Prototype"

#
# Given a dynamical system object, and a Poinare section, 
# this class implements the Poincare map.
# It return the time of time and intersection of the flow 
# with the section.
#

import DynamicModels
import PoincareSections


class PoincareMap:
    
    def __init__(self, dynsys, section):

        self.__dynsys       = dynsys
        self.__section      = section
        self.__time         = 0.0
        self.__state_vector = np.zeros(self.__dynsys.get_dim(), dtype=np.double)

    def set_time(self, time):
        self.__time = time

    def set_state_vector(self, state_vector):
        self.__state_vector = state_vector

    def go(self):
        

