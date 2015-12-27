__author__     = "Jose Rosales"
__copyright__  = "TBD"
__credits__    = "TBD"
__license__    = "TBD"
__maintainer__ = "Jose Rosales"
__email__      = "pepe.rosales@gmail.com"
__status__     = "Prototype"

__version__ = "$Revision: 1a40d4eaa00b $"
# $Source$

#System dimension
DIM = 4

#Earth mu
EARTH_MU = 398600.4418 # km^3/s^2

#Primary Body Mass (kg)
P1_MASS =  1.98855e30 

#Secondary Body Mass (kg)
P2_MASS = 5.97219e24 #7.34767309e22

# mu mass ratio parameter
MU = P2_MASS / ( P1_MASS + P2_MASS )

P1_COORD = [-MU, 0.0]
P2_COORD = [1-MU, 0.0]