# Twitter @Francesco_Turci

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as pl
import warnings
from scipy.interpolate import UnivariateSpline
from .baseeos import baseEOS

def volume_fraction(rho, sigma):
		return np.pi/6*sigma**3*rho

class PercusYevickEOS(baseEOS):
	"""Computing the Lennard-Jones chemical potential, pressure and binodal within the Percus-Yevick approximation."""
	def __init__(self, sigma, epsilon,rcut,infinity=100):
		
		self.sigma = sigma
		self.epsilon = epsilon
		self.rcut= rcut
		self.infinity = infinity*rcut

		self.integral_att = self.get_integral_att()
	
	def Vatt(self,r,rcut):
		"""Attractive part of the potential"""
		rmin = 2**(1./6.)*self.sigma 
		value = 4 *self.epsilon*( (self.sigma/r)**12-(self.sigma/r)**6)
		if r<rmin :
			return  - self.epsilon
		elif r>rcut:
			return 0
		return value


	def mu_PY(self,rho,T):
		"""Percus-Yevick approximation for the hard-sphere chemical potential."""
		eta = volume_fraction(rho,self.sigma)
		return T*(np.log(rho)+(14*eta-13*eta**2+5*eta**3)/(2*(1.0-eta)**3)-np.log(1.-eta))

	def p_PY(self,rho,T):
		"""Percus-Yevick approximation for the hard-sphere pressure."""
		eta = volume_fraction(rho,self.sigma)
		return T*rho*((1+eta+eta**2)/(1-eta)**3)

	def get_mu(self,rho,T):
		"""Hard-sphere chemical potential plus the integral over the attractive contribution."""
		return self.mu_PY(rho,T)+4*np.pi*rho*self.integral_att

	def get_p(self, rho,T):
		"""Hard-sphere pressure plus the integral over the attractive contribution."""
		return self.p_PY(rho,T)+2*np.pi*rho**2*self.integral_att



