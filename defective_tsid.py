# coding: utf8

import pinocchio as pin
import numpy as np 
import numpy.matlib as matlib
import tsid

from pinocchio.utils import *
from pinocchio import Quaternion
from pinocchio.rpy import matrixToRpy
 

## Initialization

# Definition of the tasks gains and weights
w_posture = 0.1  		# weight of the posture task
w_foot = 100.0			# weight of the feet tasks

kp_posture = 100.0  	# proportionnal gain of the posture task
kp_foot = 100.0			# proportionnal gain of the feet tasks

N_SIMULATION = 20000	# number of time steps simulated
dt = 0.001				# controller time step

t = 0.0  				# time

# Build empty lists to gather and plot the results
zHLpos = []
err_zHL = []


## Set the path where the urdf and srdf file of the robot is registered

modelPath = "/opt/openrobots/lib/python3.5/site-packages/../../../share/example-robot-data/robots"
urdf = modelPath + "/solo_description/robots/solo.urdf"
srdf = modelPath + "/solo_description/srdf/solo.srdf"
vector = pin.StdVec_StdString()
vector.extend(item for item in modelPath)
# Create the robot wrapper from the urdf model
robot = tsid.RobotWrapper(urdf, vector, pin.JointModelFreeFlyer(), False)


## Disable the gravity

robot.set_gravity_to_zero()


## Creation of the robot wrapper for the Gepetto Viewer

robot_display = pin.RobotWrapper.BuildFromURDF(urdf, [modelPath, ], pin.JointModelFreeFlyer())
robot_display.initViewer(loadModel=True)


## Take the model of the robot and load its reference configuration

model = robot.model()
pin.loadReferenceConfigurations(model, srdf, False)

# Set the current configuration q to the robot configuration straight_standing
qdes = model.referenceConfigurations['straight_standing']

# Modify the configuration from the reference one
qdes[2] = 1.0
eulerAngles = np.matrix([0.17,0.09,0.0])
quat = pin.utils.rpyToMatrix(eulerAngles.T)
for i in range(4):
	qdes[i+3] = Quaternion(quat)[i]

# Set the current velocity to zero
vdes = np.matrix(np.zeros(robot.nv)).T


## Display the robot in Gepetto Viewer

robot_display.displayCollisions(False)
robot_display.displayVisuals(True)
robot_display.display(qdes)


## Creation of the Invverse Dynamics HQP problem using the robot
## accelerations (base + joints) and the contact forces

invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
# Compute the problem data with a solver based on EiQuadProg
invdyn.computeProblemData(t, qdes, vdes)
# Get the initial data
data = invdyn.data()


## Tasks definition

# POSTURE Task
postureTask = tsid.TaskJointPosture("task-posture", robot)
postureTask.setKp(kp_posture * matlib.ones(robot.nv-6).T) # Proportional gain 
postureTask.setKd(2.0 * np.sqrt(kp_posture) * matlib.ones(robot.nv-6).T) # Derivative gain 
# Add the task to the HQP with weight = 1e-3, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(postureTask, w_posture, 1, 0.0)

# FOOT Task

# Create a mask to keep only the translation coordinates
mask = matlib.zeros(6).T
mask[2] = 1.	# mask is [0 0 1 0 0 0] so that it will keep only the translational term by z

# HL foot
HLfootTask = tsid.TaskSE3Equality("HL-foot-grounded", robot, 'HL_FOOT')
HLfootTask.setKp(kp_foot * matlib.ones(6).T)
HLfootTask.setKd(2.0 * np.sqrt(kp_foot) * matlib.ones(6).T)
HLfootTask.useLocalFrame(False)
HLfootTask.setMask(mask)
# Add the task to the HQP with weight = w_foot, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(HLfootTask, w_foot, 1, 0.0)


## TSID Trajectory

# Set the reference trajectory of the tasks

# POSTURE Task
q_ref = qdes[7:] # Initial value of the joints of the robot (in half_sitting position without the freeFlyer (6 first values))
trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)

samplePosture = trajPosture.computeNext()
postureTask.setReference(samplePosture)

# FOOT Task
pin.forwardKinematics(model, data, qdes)
	
HL_foot_ref = robot.framePosition(data, model.getFrameId('HL_FOOT'))

tHL = HL_foot_ref.translation

goals = { "HL":0.80, "HR": 0.75, "FL": 0.80, "FR": 0.75 }

HL_foot_goal = HL_foot_ref.copy()
HL_foot_goal.translation = np.matrix([tHL[0,0], tHL[1,0], goals["HL"]]).T
	
trajHLfoot = tsid.TrajectorySE3Constant("traj_HL_foot", HL_foot_goal)

sampleHLfoot = trajHLfoot.computeNext()
HLfootTask.setReference(sampleHLfoot)


## Initialisation of the solver

# Use EiquadprogFast solver
solver = tsid.SolverHQuadProgFast("qp solver")
# Resize the solver to fit the number of variables, equality and inequality constraints
solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn)


## Launch the simulation

for i in range (N_SIMULATION):
			
	pin.forwardKinematics(model, data, qdes)
	
	HL_foot_ref = robot.framePosition(data, model.getFrameId('HL_FOOT'))
	
	tHL = HL_foot_ref.translation	

	HQPData = invdyn.computeProblemData(t, qdes, vdes)
	
	sol = solver.solve(HQPData)
	
	if(sol.status != 0):
		print ("QP problem could not be solved ! Error code:", sol.status)
		break
	
	tau = invdyn.getActuatorForces(sol)
	dv = invdyn.getAccelerations(sol)
	
	vdes += dt*dv
	qdes = pin.integrate(model, qdes, dt*vdes)
	t += dt
	
	robot_display.display(qdes)
	
	## Tests/results
	
	zHLpos.append(tHL[2,0])
	
	err_zHL.append(goals["HL"] - tHL[2,0])


## Plot the results

import matplotlib.pylab as plt

ts = np.linspace(0.0, N_SIMULATION*dt, N_SIMULATION)

plt.figure(1)
plt.plot(ts, zHLpos)
plt.grid()
plt.title('Position of the HL-foot along z-axis in time')

plt.figure(2)
plt.plot(ts, err_zHL)
plt.grid()
plt.title('Position error of the HL-foot along z_axis in time')
plt.show()
