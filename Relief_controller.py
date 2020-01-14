# coding: utf8


########################################################################
#                                                                      #
#                    Control mode : tau = - D * v^                     #
#                                                                      #
########################################################################

import pinocchio as pin
import numpy as np 

pin.switchToNumpyMatrix()


########################################################################
#                    Class for a Relief Controller                     #
########################################################################

class controller:
	
	def __init__(self):
		self.error = False
	
	####################################################################
	#                      Torque Control method                       #
	####################################################################
	def control(self, qmes, vmes, t):
		
		# D Torque controller,
		D = 0.2
		tau = -D * vmes
		
		# Saturation to limit the maximal torque
		t_max = 2.5
		tau = np.maximum(np.minimum(tau, t_max * np.ones((8,1))), -t_max * np.ones((8,1)))
		
		return tau

class controller_12dof:
	
	def __init__(self):
		self.error = False
	
	####################################################################
	#                      Torque Control method                       #
	####################################################################
	def control(self, qmes12, vmes12, t):
		
		# D Torque controller,
		D = 0.2
		torques12 = -D * vmes12[6:]
		
		torques8 = np.concatenate((torques12[1:3], torques12[4:6], torques12[7:9], torques12[10:12]))
		
		# Saturation to limit the maximal torque
		t_max = 2.5
		tau = np.maximum(np.minimum(torques8, t_max * np.ones((8,1))), -t_max * np.ones((8,1)))
		
		return tau
