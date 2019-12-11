# coding: utf8

import pybullet as p 
import pinocchio as pin
import numpy as np 
import numpy.matlib as matlib
import tsid
import pybullet_data
import time

from IPython import embed 

pin.switchToNumpyMatrix()

########################################################################
#                          TSID Controller                             #
########################################################################

## Initialization

# Definition of the tasks gains and weights
w_com = 10.0			# weight of the CoM task
w_posture = 1.0  		# weight of the posture task
w_forceRef = 1e-3		# weight of the forces regularization for the contacts
w_lock = 10.0			# weight of the lock task

kp_com = 100.0 			# proportionnal gain of the CoM task
kp_posture = 10.0  		# proportionnal gain of the posture task
kd_posture = 10.0		# derivative gain of the posture task
kp_contact = 0.0		# proportionnal gain of the contacts
kp_lock = 0.0			# proportionnal gain of the lock task

# For the contacts
mu = 0.3  		# friction coefficient
fMin = 1.0		# minimum normal force
fMax = 100.0  	# maximum normal force
foot_frames = ['HL_FOOT', 'HR_FOOT', 'FL_FOOT', 'FR_FOOT']  # tab with all the foot frames names
contactNormal = np.matrix([0., 0., 1.]).T  # direction of the normal to the contact surface

# Simulation parameters
N_SIMULATION = 30000	# number of time steps simulated
dt = 0.001				# controller time step

t = 0.0  				# time

## Set the path where the urdf and srdf file of the robot is registered

modelPath = "/opt/openrobots/lib/python3.5/site-packages/../../../share/example-robot-data/robots"
urdf = modelPath + "/solo_description/robots/solo12.urdf"
srdf = modelPath + "/solo_description/srdf/solo.srdf"
vector = pin.StdVec_StdString()
vector.extend(item for item in modelPath)
# Create the robot wrapper from the urdf model
robot = tsid.RobotWrapper(urdf, vector, pin.JointModelFreeFlyer(), False)

## Creation of the robot wrapper for the Gepetto Viewer

robot_display = pin.RobotWrapper.BuildFromURDF(urdf, [modelPath, ], pin.JointModelFreeFlyer())
robot_display.initViewer(loadModel=True)


## Take the model of the robot 

model = robot.model()

## Modify the Position Bounds of the robot
model_updated = model.copy()

lowerBoundsMatrix = model.lowerPositionLimit.copy()
lowerBoundsMatrix[9] = -2.4
lowerBoundsMatrix[12] = -2.4
lowerBoundsMatrix[15] = 0.8
lowerBoundsMatrix[18] = 0.8
model_updated.lowerPositionLimit = lowerBoundsMatrix

upperBoundsMatrix = model.upperPositionLimit.copy()
upperBoundsMatrix[9] = -0.8
upperBoundsMatrix[12] = -0.8
upperBoundsMatrix[15] = 2.4
upperBoundsMatrix[18] = 2.4
model_updated.upperPositionLimit = upperBoundsMatrix

## Load its reference configuration
 
pin.loadReferenceConfigurations(model_updated, srdf, False)

# Set the current configuration q to the robot configuration straight_standing
qdes = model_updated.referenceConfigurations['straight_standing']
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

# COM Task
comTask = tsid.TaskComEquality("task-com", robot)
comTask.setKp(kp_com * matlib.ones(3).T)  # Proportional gain of the CoM task
comTask.setKd(2.0 * np.sqrt(kp_com) * matlib.ones(3).T) # Derivative gain 
# Add the task to the HQP with weight = w_com, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(comTask, w_com, 1, 0.0)

# POSTURE Task
postureTask = tsid.TaskJointPosture("task-posture", robot)
postureTask.setKp(kp_posture * matlib.ones(robot.nv).T) # Proportional gain 
postureTask.setKd(kd_posture * matlib.ones(robot.nv).T) # Derivative gain 
# Add the task to the HQP with weight = w_posture, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(postureTask, w_posture, 1, 0.0)

# LOCK Task
lockTask = tsid.TaskJointPosture("task-lock-shoulder", robot)
lockTask.setKp(kp_lock * matlib.ones(robot.nv).T) # Proportional gain 
lockTask.setKd(2.0 * np.sqrt(kp_lock) * matlib.ones(robot.nv).T) # Derivative gain
mask = np.matrix(np.zeros(robot.nv-6))
for i in [0, 3, 6, 9]:
	mask[0,i] = 1
lockTask.mask(mask.T)
# Add the task to the HQP with weight = w_lock, priority level = 0 (as real constraint) and a transition duration = 0.0
invdyn.addMotionTask(lockTask, w_lock, 0, 0.0)


## CONTACTS

contacts = 4*[None]

for i, name in enumerate(foot_frames):
	contacts[i] = tsid.ContactPoint(name, robot, name, contactNormal, mu, fMin, fMax)
	contacts[i].setKp(kp_contact * matlib.ones(3).T)
	contacts[i].setKd(2.0 * np.sqrt(kp_contact) * matlib.ones(3).T)
	H_ref = robot.framePosition(data, model_updated.getFrameId(name))
	contacts[i].setReference(H_ref)
	contacts[i].useLocalFrame(False)
	invdyn.addRigidContact(contacts[i], w_forceRef, 1.0, 1)
	

## TSID Trajectory

# Set the reference trajectory of the tasks

# COM Task
com_ref = data.com[0]  # Initial value of the CoM
trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)

sampleCom = trajCom.computeNext()
comTask.setReference(sampleCom)
	
# POSTURE Task
q_ref = qdes[7:] # Initial value of the joints of the robot (in half_sitting position without the freeFlyer (6 first values))
trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)

samplePosture = trajPosture.computeNext()
postureTask.setReference(samplePosture)

# LOCK Task
trajLock = tsid.TrajectoryEuclidianConstant("traj_lock_shoulder", q_ref)

sampleLock = trajLock.computeNext()
lockTask.setReference(sampleLock)


## Initialization of the solver

# Use EiquadprogFast solver
solver = tsid.SolverHQuadProgFast("qp solver")
# Resize the solver to fit the number of variables, equality and inequality constraints
solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn)

# Initialization of the plot variables which will be updated during the simulation loop 
# These variables describe the acceleration of the shoulders joints
acc_shoulder1 = matlib.empty((1, N_SIMULATION))
acc_shoulder2 = matlib.empty((1, N_SIMULATION))
acc_shoulder3 = matlib.empty((1, N_SIMULATION))
acc_shoulder4 = matlib.empty((1, N_SIMULATION))


########################################################################
#                              PyBullet                                #
########################################################################

# Initialization of PyBullet variables 

v_prev = np.matrix(np.zeros(robot.nv-4)).T  # velocity during the previous time step, of size (robot.nv,1)

# Start the client for PyBullet
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

# Set gravity (disabled by default)
p.setGravity(0,0,-9.81)

# Load horizontal plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

# Load Quadruped robot
robotStartPos = [0,0,0.25] 
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
p.setAdditionalSearchPath("/opt/openrobots/share/example-robot-data/robots/solo_description/robots")
robotId = p.loadURDF("solo.urdf",robotStartPos, robotStartOrientation)

# Disable default motor control for revolute joints
revoluteJointIndices = [0,1, 3,4, 6,7, 9,10]
p.setJointMotorControlArray(robotId, jointIndices = revoluteJointIndices, controlMode = p.VELOCITY_CONTROL,targetVelocities = [0.0 for m in revoluteJointIndices], forces = [0.0 for m in revoluteJointIndices])
								 
# Initialize the joint configuration
initial_joint_positions = [0.8, -1.6, 0.8, -1.6, -0.8, 1.6, -0.8, 1.6]
for i in range (len(initial_joint_positions)):
	p.resetJointState(robotId, revoluteJointIndices[i], initial_joint_positions[i])
							
# Enable torque control for revolute joints
jointTorques = [0.0 for m in revoluteJointIndices]

p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)

realTimeSimulation = True


########################################################################
#                      Torque Control function                         #
########################################################################

## Function called from the main loop which computes the inverse dynamic problem and returns the torques
def callback_torques():
	global sol, t, v_prev, q
	
	## Data collection from PyBullet
	
	jointStates = p.getJointStates(robotId, revoluteJointIndices) # State of all joints
	baseState   = p.getBasePositionAndOrientation(robotId)
	baseVel = p.getBaseVelocity(robotId)
	
	# Joint vector for Pinocchio
	q = np.vstack((np.array([baseState[0]]).transpose(), np.array([baseState[1]]).transpose(), np.array([[jointStates[i_joint][0] for i_joint in range(len(jointStates))]]).transpose()))
	v = np.vstack((np.array([baseVel[0]]).transpose(), np.array([baseVel[1]]).transpose(), np.array([[jointStates[i_joint][1] for i_joint in range(len(jointStates))]]).transpose()))
	v_dot = (v-v_prev)/dt
	v_prev = v.copy()
	
	
	## Conversion (from 8 to 12 DOF) and TSID computation 
	
	q12 = np.concatenate((q[:7], np.matrix([0.]), q[7:9], np.matrix([0.]), q[9:11], np.matrix([0.]), q[11:13], np.matrix([0.]), q[13:15]))
	v12 = np.concatenate((v[:6], np.matrix([0.]), v[6:8], np.matrix([0.]), v[8:10], np.matrix([0.]), v[10:12], np.matrix([0.]), v[12:14]))
	
	HQPData = invdyn.computeProblemData(t, q12, v12)
	
	sol = solver.solve(HQPData)
	
	tau = invdyn.getActuatorForces(sol)
	dv = invdyn.getAccelerations(sol)
	
	
	## Time incrementation and display update
	
	t += dt
	
	robot_display.display(q12)
		
		
	## Torque controller (conversion from 12 to 8 DOF)
	
	torques = np.concatenate((tau[1:3], tau[4:6], tau[7:9], tau[10:12]))
	
	
	## Saturation to limit the maximal torque
	
	t_max = 5
	torques = np.maximum(np.minimum(torques, t_max * np.ones((8,1))), -t_max * np.ones((8,1)))
	
	return torques, dv[7], dv[10], dv[13], dv[16]


########################################################################
#                             Simulator                                #
########################################################################

## Launch the simulation

for i in range (N_SIMULATION):
	
	if realTimeSimulation:
		t0 = time.clock()
	
	# Callback Pinocchio to get joint torques
	jointTorques, acc_shoulder1[:,i], acc_shoulder2[:,i], acc_shoulder3[:,i], acc_shoulder4[:,i] = callback_torques()
	
	if(sol.status != 0):
		print ("QP problem could not be solved ! Error code:", sol.status)
		break
	
	# Set control torque for all joints
	p.setJointMotorControlArray(robotId, revoluteJointIndices, controlMode=p.TORQUE_CONTROL, forces=jointTorques)
	
	# Compute one step of simulation
	p.stepSimulation()
	
	if realTimeSimulation:
		t_sleep = dt - (time.clock()-t0)
		if t_sleep > 0:
			time.sleep(t_sleep)	

########################################################################
#                              Results                                 #
########################################################################

## Plot the results

import matplotlib.pylab as plt

time = np.arange(0.0, N_SIMULATION*dt, dt)

# Acceleration tracking of the 4 shoulders
plt.figure()
plt.subplot(411)
plt.plot(time, acc_shoulder1[0,:].A1, label='acceleration shoulder 1')
plt.title('Acceleration of the 4 shoulders during the simulation')
plt.legend()
plt.subplot(412)
plt.plot(time, acc_shoulder2[0,:].A1, label='acceleration shoulder 2')
plt.legend()
plt.subplot(413)
plt.plot(time, acc_shoulder3[0,:].A1, label='acceleration shoulder 3')
plt.legend()
plt.subplot(414)
plt.plot(time, acc_shoulder4[0,:].A1, label='acceleration shoulder 4')
plt.legend()

plt.show()
