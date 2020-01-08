# coding: utf8

import pybullet as p 
import numpy as np 
import pybullet_data
import time
# import the controller class with its parameters
from PDff_controller import controller, omega, q0


########################################################################
#                        Parameters definition                         #
########################################################################
	
# Simulation parameters
N_SIMULATION = 10000	# number of time steps simulated
dt = 0.001				# controller time step

t = 0.0  				# time

# Set the simulation in real time
realTimeSimulation = True


########################################################################
#                              PyBullet                                #
########################################################################

# Start the client for PyBullet
physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version

# Load horizontal plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Load Quadruped robot
robotStartPos = [0,0,0.5] 
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
p.setAdditionalSearchPath("/opt/openrobots/share/example-robot-data/robots/solo_description/robots")
robotId = p.loadURDF("solo.urdf",robotStartPos, robotStartOrientation)

# Disable default motor control for revolute joints
revoluteJointIndices = [0,1, 3,4, 6,7, 9,10]
p.setJointMotorControlArray(robotId, jointIndices = revoluteJointIndices, controlMode = p.VELOCITY_CONTROL,targetVelocities = [0.0 for m in revoluteJointIndices], forces = [0.0 for m in revoluteJointIndices])

# Initialize the robot in a specific configuration
p.resetJointStatesMultiDof(robotId, revoluteJointIndices, q0)						

# Enable torque control for revolute joints
jointTorques = [0.0 for m in revoluteJointIndices]

p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)

# Fix the base in the world frame
p.createConstraint(robotId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0.5])

# Set time step for the simulation
p.setTimeStep(dt)


########################################################################
#                             Simulator                                #
########################################################################

myController = controller(q0, omega)

Qdes = []
Qmes = []
Vdes = []
Vmes = []
tau0 = []
t_list = []

for i in range (N_SIMULATION):
	
	if realTimeSimulation:	
		time_start = time.time()
		
	####################################################################
	#                 Data collection from PyBullet                    #
	####################################################################
	
	jointStates = p.getJointStates(robotId, revoluteJointIndices) # State of all joints
	
	# Joints configuration and velocity vector
	qmes = np.vstack((np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).T))
	vmes = np.vstack((np.array([[jointStates[i_joint][1] for i_joint in range(len(jointStates))]]).T))
	
	####################################################################
	#         Retrieve the joint torques from the controller           #
	####################################################################
	
	jointTorques = myController.control(qmes, vmes, t)
	
	# Stop the simulation if there is an error with the controller
	if(myController.error):
		print ("Error ! Diverging results")
		break
	
	# Set control torque for all joints
	p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)
	
	# Tracking of the trajectories
	tau0.append(jointTorques[0])
	"""Qdes.append(myController.qdes[0].copy())
	Vdes.append(myController.vdes[0].copy())
	Qmes.append(qmes[0])
	Vmes.append(vmes[0])"""
	
	# Compute one step of simulation
	p.stepSimulation()
	
	# Time incrementation
	t += dt
	
	if realTimeSimulation:
		time_spent = time.time() - time_start
		if time_spent < dt:
			time.sleep(dt-time_spent)	# ensure the simulation runs in real time
	
	t_list.append(time_spent)
			

## Plot the tracking of the trajectories

import matplotlib.pylab as plt
"""
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(Qdes, 'b', label="Desired position")
plt.plot(Qmes, 'r', label="Measured position")
plt.grid()
plt.legend()
plt.title("Trajectories tracking")
plt.subplot(2,1,2)
plt.plot(Vdes, 'b', label="Desired velocity")
plt.plot(Vmes, 'r', label="Measured velocity")
plt.grid()
plt.legend()
plt.show()
plt.figure(1)
plt.plot(tau0)
plt.show()"""
plt.figure(2)
plt.plot(t_list, 'k+')
plt.show()
