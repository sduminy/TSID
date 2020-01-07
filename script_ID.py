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
#                   Class for a sinus Controller                       #
########################################################################

class sinController:
	
	def __init__(self, q0, omega):
		self.omega = omega
		self.qdes = q0.copy()
		self.vdes = np.zeros((8,1))
		self.ades = np.zeros((8,1))
		
	####################################################################
	#                      Torque Control method                       #
	####################################################################
	def control(self, qmes, vmes, t):
		# Definition of qdes, vdes and ades
		self.qdes[0] = np.sin(self.omega * t)
		self.vdes[0] = self.omega * np.cos(self.omega * t)
		self.ades[0] = -self.omega**2 * np.sin(self.omega * t)
		#print(self.qdes[0], self.vdes[0])
		
		# Definition of the solver
		HQPData = invdyn.computeProblemData(t, self.qdes, self.vdes)
		self.sol = solver.solve(HQPData)
		
		# PD Torque controller
		P = 10.0
		D = 2.0 * np.sqrt(P)/2
		tau = P * (self.qdes - qmes) - D * vmes
		
		# Saturation to limit the maximal torque
		t_max = 2.5
		tau = np.maximum(np.minimum(tau, t_max * np.ones((8,1))), -t_max * np.ones((8,1)))
	
		return tau[0]


########################################################################
#                        Parameters definition                         #
########################################################################

# Parameters of the desired trajectory
T = 2 * np.pi					# sinus period (s)
omega = 2 * np.pi / T	# sinus pulsation
 
# Simulation parameters
N_SIMULATION = 10000	# number of time steps simulated
dt = 0.001				# controller time step

t = 0.0  				# time

# Set the simulation in real time
realTimeSimulation = True


########################################################################
#             Definition of the Model and TSID problem                 #
########################################################################

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

# Set time step for the simulation
p.setTimeStep(dt)


########################################################################
#                             Simulator                                #
########################################################################

myController = sinController(q0, omega)

Qdes = []
Qmes = []
Vdes = []
Vmes = []

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
	
	jointTorques[0] = myController.control(qmes, vmes, t)
	print(myController.qdes[0], myController.vdes[0])
	"""qdes0 = myController.qdes[0]
	vdes0 = myController.vdes[0]
	print(qdes0, vdes0)"""
	# Stop the simulation if the QP problem can't be solved
	if(myController.sol.status != 0):
		print ("QP problem could not be solved ! Error code:", myController.sol.status)
		break
	
	# Set control torque for all joints
	p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)
	
	# Tracking of the trajectories
	Qdes.append(myController.qdes[0].copy())
	Vdes.append(myController.vdes[0].copy())
	Qmes.append(qmes[0])
	Vmes.append(vmes[0])
	
	# Compute one step of simulation
	p.stepSimulation()
	
	# Time incrementation
	t += dt
	
	if realTimeSimulation:
		time_spent = time.time() - time_start
		if time_spent < dt:
			time.sleep(dt-time_spent)	# ensure the simulation runs in real time
			

## Plot the tracking of the trajectories

import matplotlib.pylab as plt
plt.figure()
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
