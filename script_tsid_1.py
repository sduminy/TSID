# coding: utf8


########################################################################
#                                                                      #
#            Loi de commande : tau = P(q^-q*) + D(v^)                  #
#                                                                      #
########################################################################


import pybullet as p 
import pinocchio as pin
import numpy as np 
import tsid
import pybullet_data
import time

pin.switchToNumpyMatrix()


########################################################################
#                          TSID Controller                             #
########################################################################

## Initialization

# Parameters of the configuration test
f = 1					# sinus frequency (Hz)
omega = 2 * np.pi * f	# sinus pulsation
 
# Simulation parameters
N_SIMULATION = 20000	# number of time steps simulated
dt = 0.001				# controller time step

t = 0.0  				# time

# Set the simulation in real time
realTimeSimulation = True


## Set the paths where the urdf and srdf file of the robot are registered

modelPath = "/opt/openrobots/lib/python3.5/site-packages/../../../share/example-robot-data/robots"
urdf = modelPath + "/solo_description/robots/solo.urdf"
srdf = modelPath + "/solo_description/srdf/solo.srdf"
vector = pin.StdVec_StdString()
vector.extend(item for item in modelPath)


## Create the robot wrapper from the urdf model (without the free flyer)

robot = tsid.RobotWrapper(urdf, vector, False)


## Take the model of the robot and load its reference configuration

model = robot.model()
pin.loadReferenceConfigurations(model, srdf, False)


## Set the initial configuration vector to the robot configuration straight_standing
## And set the initial velocity to zero

q0 = np.zeros((robot.nq,1))
v0 = np.zeros((robot.nv,1))

## Creation of the Invverse Dynamics HQP problem using the robot
## accelerations (base + joints) and the contact forces

invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
# Compute the problem data with a solver based on EiQuadProg
invdyn.computeProblemData(t, q0, v0)
# Get the initial data
data = invdyn.data()


## Initialization of the solver

# Use EiquadprogFast solver
solver = tsid.SolverHQuadProgFast("qp solver")
# Resize the solver to fit the number of variables, equality and inequality constraints
solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn)


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
#p.resetJointState(robotId, revoluteJointIndices[0], 0.2)						

# Enable torque control for revolute joints
jointTorques = [0.0 for m in revoluteJointIndices]

p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)

# Fix the base in the world frame
p.createConstraint(robotId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0.5])
#p.createConstraint(robotId, p.getJointInfo(robotId,1)[16], robotId, p.getJointInfo(robotId,1)[0], p.JOINT_FIXED, p.getJointInfo(robotId,1)[13], p.getJointInfo(robotId,1)[14], [0, -1, 0])

# Set time step for the simulation
p.setTimeStep(dt)


########################################################################
#                      Torque Control function                         #
########################################################################

## Function called from the main loop which computes the inverse dynamic problem and returns the torques
def callback_torques():
	
	global sol, t, q0	# variables needed/computed during the simulation
	
	## Data collection from PyBullet
	
	jointStates = p.getJointStates(robotId, revoluteJointIndices) # State of all joints
	
	# Joints configuration and velocity vector
	qmes = np.vstack((np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).T))
	vmes = np.vstack((np.array([[jointStates[i_joint][1] for i_joint in range(len(jointStates))]]).T))
	
	## Definition of qdes and vdes
	
	qdes = q0.copy()
	qdes[1] = np.sin(omega*t) + q0[1]
	vdes = v0.copy()
	vdes[1] = omega * np.cos(omega*t)
	accdes = v0.copy()
	accdes[1] = -omega**2 * np.sin(omega*t)
	
	
	## Resolution of the HQP problem
	
	HQPData = invdyn.computeProblemData(t, qdes, vdes)
	
	sol = solver.solve(HQPData)
	
	
	## Integration of the tsid solution
	
	#acc = invdyn.getAccelerations(sol)
	vdes += dt * accdes
	qdes = pin.integrate(model, qdes, vdes*dt)


	## Time incrementation
	
	t += dt
	
				
	## Torque PD controller 
	
	P = 1.0
	D = 2 * np.sqrt(P)
	torques = P * (qmes - qdes) + D * vmes 
	
		
	## Saturation to limit the maximal torque
	
	t_max = 2.5
	torques = np.maximum(np.minimum(torques, t_max * np.ones((8,1))), -t_max * np.ones((8,1)))
	
	return torques[1]


########################################################################
#                             Simulator                                #
########################################################################

## Launch the simulation

t_list = []	# list to verify that each iteration of the simulation is less than 1 ms

for i in range (N_SIMULATION):
	
	if realTimeSimulation:	
		time_start = time.time()
	
	# Callback Pinocchio to get joint torques
	jointTorques[1] = callback_torques()
	
	# Stop the simulation if the QP problem can't be solved
	if(sol.status != 0):
		print ("QP problem could not be solved ! Error code:", sol.status)
		break
	
	# Set control torque for all joints
	p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)
	
	# Compute one step of simulation
	p.stepSimulation()
	
	if realTimeSimulation:
		time_spent = time.time() - time_start
		if time_spent < dt:
			time.sleep(dt-time_spent)	# ensure the simulation runs in real time
			
	t_list.append(time_spent)

## Plot the list of the duration of each iteration
## It should be less than 0.001 s

import matplotlib.pylab as plt

plt.plot(t_list, '+k')
plt.grid()
plt.show()
