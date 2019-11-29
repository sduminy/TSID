# coding: utf8

import pinocchio as pin
import numpy as np 
import numpy.matlib as matlib
import tsid

from pinocchio.utils import *
from pinocchio import Quaternion
from pinocchio.rpy import matrixToRpy

from IPython import embed 

## Initialization

# Definition of the tasks gains and weights
w_posture = 1.0  		# weight of the posture task
w_foot = 100.0			# weight of the feet tasks

kp_posture = 1.0  	# proportionnal gain of the posture task
kp_foot = 100.0			# proportionnal gain of the feet tasks

N_SIMULATION = 20000	# number of time steps simulated
dt = 0.001				# controller time step

t = 0.0  				# time

# Build empty lists to gather and plot the results
zFRpos = []
zFLpos = []
zHRpos = []
zHLpos = []
err_zFR = []
err_zFL = []
err_zHR = []
err_zHL = []


## Set the path where the urdf and srdf file of the robot is registered

modelPath = "/opt/openrobots/lib/python3.5/site-packages/../../../share/example-robot-data/robots"
urdf = modelPath + "/solo_description/robots/solo.urdf"
srdf = modelPath + "/solo_description/srdf/solo.srdf"
vector = pin.StdVec_StdString()
vector.extend(item for item in modelPath)
# Create the robot wrapper from the urdf model
robot = tsid.RobotWrapper(urdf, vector, False)


## Disable the gravity

robot.set_gravity_to_zero()


## Creation of the robot wrapper for the Gepetto Viewer

robot_display = pin.RobotWrapper.BuildFromURDF(urdf, [modelPath, ])
robot_display.initViewer(loadModel=True)


## Take the model of the robot and load its reference configuration

model = robot.model()
pin.loadReferenceConfigurations(model, srdf, False)

# Set the current configuration q to the robot configuration straight_standing
qdes = model.referenceConfigurations['straight_standing']

"""
# Modify the configuration from the reference one
qdes[2] = 1.0
eulerAngles = np.matrix([0.17,0.09,0.0])
quat = pin.utils.rpyToMatrix(eulerAngles.T)
for i in range(4):
	qdes[i+3] = Quaternion(quat)[i]
"""
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
postureTask.setKp(kp_posture * matlib.ones(robot.nv).T) # Proportional gain 
postureTask.setKd(2.0 * np.sqrt(kp_posture) * matlib.ones(robot.nv).T) # Derivative gain 
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

# HR foot
HRfootTask = tsid.TaskSE3Equality("HR-foot-grounded", robot, 'HR_FOOT')
HRfootTask.setKp(kp_foot * matlib.ones(6).T)
HRfootTask.setKd(2.0 * np.sqrt(kp_foot) * matlib.ones(6).T)
HRfootTask.useLocalFrame(False)
HRfootTask.setMask(mask)
# Add the task to the HQP with weight = 1.0, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(HRfootTask, w_foot, 1, 0.0)

# FL foot
FLfootTask = tsid.TaskSE3Equality("FL-foot-grounded", robot, 'FL_FOOT')
FLfootTask.setKp(kp_foot * matlib.ones(6).T)
FLfootTask.setKd(2.0 * np.sqrt(kp_foot) * matlib.ones(6).T)
FLfootTask.useLocalFrame(False)
FLfootTask.setMask(mask)
# Add the task to the HQP with weight = 1.0, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(FLfootTask, w_foot, 1, 0.0)

# FR foot
FRfootTask = tsid.TaskSE3Equality("FR-foot-grounded", robot, 'FR_FOOT')
FRfootTask.setKp(kp_foot * matlib.ones(6).T)
FRfootTask.setKd(2.0 * np.sqrt(kp_foot) * matlib.ones(6).T)
FRfootTask.useLocalFrame(False)
FRfootTask.setMask(mask)
# Add the task to the HQP with weight = 1.0, priority level = 0 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(FRfootTask, w_foot, 1, 0.0)


## TSID Trajectory

# Set the reference trajectory of the tasks

# POSTURE Task
q_ref = qdes # Initial value of the joints of the robot (in half_sitting position without the freeFlyer (6 first values))
trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)

samplePosture = trajPosture.computeNext()
postureTask.setReference(samplePosture)

# FOOT Task
pin.forwardKinematics(model, data, qdes)
	
HL_foot_ref = robot.framePosition(data, model.getFrameId('HL_FOOT'))
HR_foot_ref = robot.framePosition(data, model.getFrameId('HR_FOOT'))
FL_foot_ref = robot.framePosition(data, model.getFrameId('FL_FOOT'))
FR_foot_ref = robot.framePosition(data, model.getFrameId('FR_FOOT'))

tHL = HL_foot_ref.translation
tHR = HR_foot_ref.translation
tFL = FL_foot_ref.translation
tFR = FR_foot_ref.translation
	
goals = { "HL":-0.15, "HR": -0.15, "FL": -0.15, "FR": -0.15 }

HL_foot_goal = HL_foot_ref.copy()
HL_foot_goal.translation = np.matrix([tHL[0,0], tHL[1,0], goals["HL"]]).T

FR_foot_goal = FR_foot_ref.copy()
FR_foot_goal.translation = np.matrix([tFR[0,0], tFR[1,0], goals["FR"]]).T
  
FL_foot_goal = FL_foot_ref.copy()
FL_foot_goal.translation = np.matrix([tFL[0,0], tFL[1,0], goals["FL"]]).T
  
HR_foot_goal = HR_foot_ref.copy()
HR_foot_goal.translation = np.matrix([tHR[0,0], tHR[1,0], goals["HR"]]).T
	
trajHLfoot = tsid.TrajectorySE3Constant("traj_HL_foot", HL_foot_goal)

trajHRfoot = tsid.TrajectorySE3Constant("traj_HR_foot", HR_foot_goal)
	
trajFLfoot = tsid.TrajectorySE3Constant("traj_FL_foot", FL_foot_goal)
	
trajFRfoot = tsid.TrajectorySE3Constant("traj_FR_foot", FR_foot_goal)

sampleHLfoot = trajHLfoot.computeNext()
HLfootTask.setReference(sampleHLfoot)

sampleHRfoot = trajHRfoot.computeNext()
HRfootTask.setReference(sampleHRfoot)

sampleFLfoot = trajFLfoot.computeNext()
FLfootTask.setReference(sampleFLfoot)

sampleFRfoot = trajFRfoot.computeNext()
FRfootTask.setReference(sampleFRfoot)


## Initialisation of the solver

# Use EiquadprogFast solver
solver = tsid.SolverHQuadProgFast("qp solver")
# Resize the solver to fit the number of variables, equality and inequality constraints
solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn)


## Launch the simulation

for i in range (N_SIMULATION):
			
	pin.forwardKinematics(model, data, qdes)
	
	HL_foot_ref = robot.framePosition(data, model.getFrameId('HL_FOOT'))
	HR_foot_ref = robot.framePosition(data, model.getFrameId('HR_FOOT'))
	FL_foot_ref = robot.framePosition(data, model.getFrameId('FL_FOOT'))
	FR_foot_ref = robot.framePosition(data, model.getFrameId('FR_FOOT'))

	tHL = HL_foot_ref.translation	
	tHR = HR_foot_ref.translation
	tFL = FL_foot_ref.translation
	tFR = FR_foot_ref.translation
	
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
	
	zFRpos.append(tFR[2,0])
	zFLpos.append(tFL[2,0])
	zHRpos.append(tHR[2,0])
	zHLpos.append(tHL[2,0])
	
	err_zFR.append(goals["FR"] - tFR[2,0])
	err_zFL.append(goals["FL"] - tFL[2,0])
	err_zHR.append(goals["HR"] - tHR[2,0])
	err_zHL.append(goals["HL"] - tHL[2,0])
	

## Plot the results

import matplotlib.pylab as plt

ts = np.linspace(0, N_SIMULATION*dt, N_SIMULATION)

plt.figure(1)
plt.plot(ts, zFRpos, label='FR foot')
plt.plot(ts, zFLpos, label='FL foot')
plt.plot(ts, zHRpos, label='HR foot')
plt.plot(ts, zHLpos, label='HL foot')
plt.grid()
plt.title('Position of each foot along z-axis function of time')
plt.legend()

plt.figure(2)
plt.plot(ts, err_zFR, label='FR foot')
plt.plot(ts, err_zFL, label='FL foot')
plt.plot(ts, err_zHR, label='HR foot')
plt.plot(ts, err_zHL, label='HL foot')
plt.grid()
plt.title('Position errors of each foot along z_axis function of time')
plt.legend()
plt.show()

