# coding: utf8


########################################################################
#                                                                      #
#            Loi de commande : tau = P(q*-q^) - D(v^)                  #
#                                                                      #
########################################################################

import pinocchio as pin
import numpy as np 

pin.switchToNumpyMatrix()


########################################################################
#                Class for a proportionnal Controller                  #
########################################################################

class controller:
	
	def __init__(self, q0, omega):
		self.omega = omega
		self.q0 = q0
		self.qdes = q0.copy()
		self.vdes = np.zeros((8,1))
		self.ades = np.zeros((8,1))
		self.error = False
		
	####################################################################
	#                      Torque Control method                       #
	####################################################################
	def control(self, qmes, vmes, t):
		# Definition of qdes, vdes and ades
		self.qdes = np.sin(self.omega * t) + self.q0
		self.vdes = self.omega * np.cos(self.omega * t)
		self.ades = -self.omega**2 * np.sin(self.omega * t)
		
		# PD Torque controller
		P = np.diag((10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0))
		D = np.diag((3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0))
		tau = np.array(np.matrix(np.diag(P * (self.qdes - qmes) - D * vmes)).T)
		
		# Saturation to limit the maximal torque
		t_max = 2.5
		tau = np.maximum(np.minimum(tau, t_max * np.ones((8,1))), -t_max * np.ones((8,1)))
		
		self.error = np.linalg.norm(vmes)>1000
		
		return tau

# Parameters of the desired trajectory

omega = np.zeros((8,1))		# sinus pulsation

q0 = np.ones((8,1))		# initial configuration

for i in range(8):
	omega[i] = 1.0
	q0[i] = 0.2
