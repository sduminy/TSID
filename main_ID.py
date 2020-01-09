# coding: utf8

import pybullet as p 
import numpy as np 
import pybullet_data
import time
# import the controller class with its parameters
from PDff_controller import controller, omega, q0
import Relief_controller
import EmergencyStop_controller

########################################################################
#                        Parameters definition                         #
########################################################################
	
# Simulation parameters
N_SIMULATION = 10000	# number of time steps simulated
dt = 0.001				# controller time step

t = 0.0  				# time

# Set the simulation in real time
realTimeSimulation = True

# Initialize the error for the simulation time
time_error = False


########################################################################
#                              PyBullet                                #
########################################################################

# Start the client for PyBullet
physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version

# Load horizontal plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Set the gravity
p.setGravity(0,0,-9.81)

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
myReliefController = Relief_controller.controller()
myEmergencyStop = EmergencyStop_controller.controller()

Qdes = [[],[],[],[],[],[],[],[]]
Vdes = [[],[],[],[],[],[],[],[]]
Qmes = [[],[],[],[],[],[],[],[]]
Vmes = [[],[],[],[],[],[],[],[]]
Tau = [[],[],[],[],[],[],[],[]]

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
	
	if(myController.error):
		myController = myReliefController
	
	jointTorques = myController.control(qmes, vmes, t)
	
	# Stop the simulation if there is an error with the controller or the simulation time
	
	time_error = time_error or (time.time()-time_start > 0.003)
	if (time_error):
		myController = myEmergencyStop
		
	
	# Set control torque for all joints
	p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)
	
	# Tracking of the trajectories
	
	for i in range(8):
		"""Qdes[i].append(myController.qdes[i].copy())
		Vdes[i].append(myController.vdes[i].copy())"""
		Qmes[i].append(qmes[i])
		Vmes[i].append(vmes[i])
		Tau[i].append(jointTorques[i])
	
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

plt.figure(1)

plt.subplot(3,1,1)
for i in range(8):
	plt.plot(Qdes[i], '-')
	plt.plot(Qmes[i], '--')
plt.grid()
plt.title("Configuration tracking")

plt.subplot(3,1,2)
for i in range(8):
	plt.plot(Vdes[i], '-')
	plt.plot(Vmes[i], '--')
plt.grid()
plt.title("Velocity tracking")

plt.subplot(3,1,3)

for i in range(8):
	plt.plot(Tau[i], '-')
plt.grid()
plt.title("Torques tracking")

plt.show()	

plt.figure(2)
plt.plot(t_list, 'k+')

plt.show()
