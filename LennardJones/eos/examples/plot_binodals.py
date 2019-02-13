#Twitter @Francesco_Turci

#In this example we plot the sbinodal line for a Lennard-Jones system with cutoff 2.5 sigma with three different equations of state:
# (1) with the Percus-Yevick approximation
# (2) with the Carnahan-Starling approximation
# (3) with the Kolafa-Nezbevda parametric fit to the Molecular Dynamics and Monte Carlo data

import pylab as pl
from liquidpy.eos.LennardJones import percus_yevick as py
from liquidpy.eos.LennardJones import carnahan_starling as cs
from liquidpy.eos.LennardJones import kolafa_nezbevda as kn


epsilon = 1.0
sigma = 1.0
rcut = 2.5
PY = py.PercusYevickEOS(sigma, epsilon,rcut)
CS = cs.CarnahanStarlingEOS(sigma, epsilon,rcut)
# Kolafa Nezbevda assumes sigma=1 and epsilon=1
KN = kn.KolafaNezbevdaEOS()

models = [PY, CS, KN]
color = ['k','r','g']
Tmax = [1.32, 1.33,1.34]
label = ["Percus-Yevick", "Carnahan-Starling", "Kolafa-Nezbevda"]

for i,model in enumerate(models):
	# compute the coexistence
	model.coexistence(Tmax[i])
	model.plot_binodal(show=False,color=color[i], label=label[i])
pl.legend(frameon=False)
pl.show()