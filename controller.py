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
		self.qdes[0] = np.sin(self.omega * t)
		self.vdes[0] = self.omega * np.cos(self.omega * t)
		self.ades[0] = -self.omega**2 * np.sin(self.omega * t)
		
		# PD Torque controller
		P = 10.0
		D = 2.0 * np.sqrt(P)/2
		tau = P * (self.qdes - qmes) - D * vmes
		
		# Saturation to limit the maximal torque
		t_max = 2.5
		tau = np.maximum(np.minimum(tau, t_max * np.ones((8,1))), -t_max * np.ones((8,1)))
		
		self.error = np.linalg.norm(vmes)>1000
		
		return tau[0]

# Parameters of the desired trajectory
T = 2 * np.pi			# sinus period (s)
omega = 2 * np.pi / T	# sinus pulsation
q0 = np.zeros((8,1))
