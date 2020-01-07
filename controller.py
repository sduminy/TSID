# coding: utf8


########################################################################
#                                                                      #
#            Loi de commande : tau = P(q^-q*) + D(v^)                  #
#                                                                      #
########################################################################

import pinocchio as pin
import numpy as np 

pin.switchToNumpyMatrix()


########################################################################
#                Class for a proportionnal Controller                  #
########################################################################

class P_controller:
	
	def __init__(self, q0, omega):
		self.omega = omega
		self.qdes = q0.copy()
		self.vdes = np.zeros((8,1))
		self.ades = np.zeros((8,1))
		self.error = False
		
	####################################################################
	#                      Torque Control method                       #
	####################################################################
	def control(self, qmes, vmes, t):
		# Definition of qdes, vdes and ades
		self.qdes = np.sin(self.omega * t) + q0
		self.vdes = self.omega * np.cos(self.omega * t)
		self.ades = -self.omega**2 * np.sin(self.omega * t)
		
		# PD Torque controller
		P = np.diag((10.0, 0, 0, 0, 0, 0, 0, 0))
		D = np.diag((3.0, 0, 0, 0, 0, 0, 0, 0))
		tau = np.array(np.matrix(np.diag(P * (self.qdes - qmes) - D * vmes)).T)
		
		# Saturation to limit the maximal torque
		t_max = 2.5
		tau = np.maximum(np.minimum(tau, t_max * np.ones((8,1))), -t_max * np.ones((8,1)))
		
		self.error = np.linalg.norm(vmes)>1000
		
		return tau

# Parameters of the desired trajectory

T = 2 * np.pi				# sinus period (s)

omega = np.zeros((8,1))		# sinus pulsation
omega[0] = 2 * np.pi / T	

q0 = np.zeros((8,1))		# initial configuration
