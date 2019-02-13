# Twitter  @Francesco_Turci

# Compute the effective radius of a particle with Weeks-Chandler-Anderson interactions at a given temperature

import numpy as np
import pylab as pl
import scipy.integrate as integrate

from scipy.optimize import curve_fit

epsilon=1

def LJ(r, sigma=1 ):
	return 4*epsilon*((sigma/r)**12-(sigma/r)**6)

def WCA(r,rmin=2**(1/6.)):
	return (LJ(r)-LJ(rmin))*np.heaviside(rmin-r,0.0)

def diameter(T, u=WCA, rmin=0, rmax=2**(1/6.)):
	beta = 1./T

	def integrand(r):
		return 1-np.exp(-beta*u(r))

	return integrate.quad(integrand,rmin,rmax)


def fitter(x,a,b,c,d,e):
	return a+b*np.log(x)+c*x+d*x**2+e*x**3

def test():
	Ts = np.linspace(0.01, 10,1000)
	diameters = np.array([ diameter(T) for T in Ts])

	popt,pcov= curve_fit(fitter,Ts,diameters[:,0])

	pl.plot(Ts,diameters[:,0],'o', alpha=0.1)
	pl.plot(Ts,fitter(Ts,*popt))
	pl.xlabel(r"$r$")
	pl.ylabel(r"$diameter/\sigma$")

	# pl.plot(Ts , popt[0]- 1./12.*np.log(Ts/epsilon),'k')
	pl.show()


