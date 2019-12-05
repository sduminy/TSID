# coding: utf8

import pinocchio as pin
import numpy as np 
import numpy.matlib as matlib
import tsid
import time

from IPython import embed 

pin.switchToNumpyMatrix()

## Initialization

# Definition of the tasks gains and weights
w_com = 10.0				# weight of the CoM task
w_posture = 1.0  		# weight of the posture task
w_forceRef = 1e-3		# weight of the forces regularization for the contacts

kp_com = 10.0 			# proportionnal gain of the CoM task
kp_posture = 0.0  		# proportionnal gain of the posture task
kd_posture = 1.0		# derivative gain of the posture task
kp_contact = 10.0		# proportionnal gain of the contacts

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


## Take the model of the robot and load its reference configuration

model = robot.model()
pin.loadReferenceConfigurations(model, srdf, False)

# Set the current configuration q to the robot configuration straight_standing
qdes = model.referenceConfigurations['straight_standing']

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

# COM Task
comTask = tsid.TaskComEquality("task-com", robot)
comTask.setKp(kp_com * matlib.ones(3).T)  # Proportional gain of the CoM task
comTask.setKd(2.0 * np.sqrt(kp_com) * matlib.ones(3).T) # Derivative gain 
# Add the task to the HQP with weight = 1.0, priority level = 0 (as real constraint) and a transition duration = 0.0
invdyn.addMotionTask(comTask, w_com, 1, 0.0)

# POSTURE Task
postureTask = tsid.TaskJointPosture("task-posture", robot)
postureTask.setKp(kp_posture * matlib.ones(robot.nv).T) # Proportional gain 
postureTask.setKd(kd_posture * matlib.ones(robot.nv).T) # Derivative gain 
# Add the task to the HQP with weight = 1e-3, priority level = 1 (in the cost function) and a transition duration = 0.0
invdyn.addMotionTask(postureTask, w_posture, 1, 0.0)


## CONTACTS

contacts = 4*[None]

for i, name in enumerate(foot_frames):
	contacts[i] = tsid.ContactPoint(name, robot, name, contactNormal, mu, fMin, fMax)
	contacts[i].setKp(kp_contact * matlib.ones(3).T)
	contacts[i].setKd(2.0 * np.sqrt(kp_contact) * matlib.ones(3).T)
	H_ref = robot.framePosition(data, model.getFrameId(name))
	contacts[i].setReference(H_ref)
	contacts[i].useLocalFrame(False)
	invdyn.addRigidContact(contacts[i], w_forceRef)
	

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

## Initialisation of the solver

# Use EiquadprogFast solver
solver = tsid.SolverHQuadProgFast("qp solver")
# Resize the solver to fit the number of variables, equality and inequality constraints
solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn)

# Initialisation of the plot variables which will be updated during the simulation loop 
# These variables describe the behavior of the CoM of the robot (reference and real position, velocity and acceleration)
com_pos = matlib.empty((3, N_SIMULATION))
com_vel = matlib.empty((3, N_SIMULATION))
com_acc = matlib.empty((3, N_SIMULATION))

com_pos_ref = matlib.empty((3, N_SIMULATION))
com_vel_ref = matlib.empty((3, N_SIMULATION))
com_acc_ref = matlib.empty((3, N_SIMULATION))
com_acc_des = matlib.empty((3, N_SIMULATION))

## Launch the simulation
simulation_time = []
for i in range (N_SIMULATION):
	
	#time_start = time.time()
	
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
	
	#simulation_time.append(time.time() - time_start)
	
	com_pos[:,i] = robot.com(invdyn.data())
	com_vel[:,i] = robot.com_vel(invdyn.data())
	com_acc[:,i] = comTask.getAcceleration(dv)
	com_pos_ref[:,i] = sampleCom.pos()
	com_vel_ref[:,i] = sampleCom.vel()
	com_acc_ref[:,i] = sampleCom.acc()
	com_acc_des[:,i] = comTask.getDesiredAcceleration	
	
embed()

## Plot the results

import matplotlib.pylab as plt

time = np.arange(0.0, N_SIMULATION*dt, dt)

# Position tracking of the CoM along the x,y,z axis
plt.figure(1)
plt.subplot(311)
plt.plot(time, com_pos[0,:].A1, label='CoM x')
plt.plot(time, com_pos_ref[0,:].A1, 'r:', label='CoM Ref x')
plt.title('Position error of the Center of Mass task')
plt.legend()
plt.subplot(312)
plt.plot(time, com_pos[1,:].A1, label='CoM y')
plt.plot(time, com_pos_ref[1,:].A1, 'r:', label='CoM Ref y')
plt.legend()
plt.subplot(313)
plt.plot(time, com_pos[2,:].A1, label='CoM z')
plt.plot(time, com_pos_ref[2,:].A1, 'r:', label='CoM Ref z')
plt.legend()

# Velocity tracking of the CoM along the x,y,z axis
plt.figure(2)
plt.subplot(311)
plt.plot(time, com_vel[0,:].A1, label='CoM x')
plt.plot(time, com_vel_ref[0,:].A1, 'r:', label='CoM Ref x')
plt.title('Velocity error of the Center of Mass task')
plt.legend()
plt.subplot(312)
plt.plot(time, com_vel[1,:].A1, label='CoM y')
plt.plot(time, com_vel_ref[1,:].A1, 'r:', label='CoM Ref y')
plt.legend()
plt.subplot(313)
plt.plot(time, com_vel[2,:].A1, label='CoM z')
plt.plot(time, com_vel_ref[2,:].A1, 'r:', label='CoM Ref z')
plt.legend()

# Acceleration tracking of the CoM along the x,y,z axis
plt.figure(3)
plt.subplot(311)
plt.plot(time, com_acc[0,:].A1, label='CoM x')
plt.plot(time, com_acc_ref[0,:].A1, 'r:', label='CoM Ref x')
plt.plot(time, com_acc_des[0,:].A1, 'g--', label='CoM Des x')
plt.title('Acceleration error of the Center of Mass task')
plt.legend()
plt.subplot(312)
plt.plot(time, com_acc[1,:].A1, label='CoM y')
plt.plot(time, com_acc_ref[1,:].A1, 'r:', label='CoM Ref y')
plt.plot(time, com_acc_des[1,:].A1, 'g--', label='CoM Des y')
plt.legend()
plt.subplot(313)
plt.plot(time, com_acc[2,:].A1, label='CoM z')
plt.plot(time, com_acc_ref[2,:].A1, 'r:', label='CoM Ref z')
plt.plot(time, com_acc_des[2,:].A1, 'g--', label='CoM Des z')
plt.legend()

plt.show()
