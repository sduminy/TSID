# coding: utf8


########################################################################
#                                                                      #
#         			  Control law : tau = tau_TSID                     #
#                                                                      #
########################################################################

import pinocchio as pin
import numpy as np
import numpy.matlib as matlib
import tsid

pin.switchToNumpyMatrix()


########################################################################
#            Class for a PD with feed-forward Controller               #
########################################################################

class controller:
	
	def __init__(self, q0, t):
		
		self.qdes = q0.copy()
		self.vdes = np.zeros((8,1))
		self.ades = np.zeros((8,1))
		self.error = False
		
		kp_foot = 10.0
		w_foot = 1.0
		kp_posture = 10.0
		w_posture = 1.0
		
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

		self.robot = tsid.RobotWrapper(urdf, vector, False)
		
		self.model = self.robot.model()
		
		## Creation of the Invverse Dynamics HQP problem using the robot
		## accelerations (base + joints) and the contact forces

		self.invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", self.robot, False)
		# Compute the problem data with a solver based on EiQuadProg
		self.invdyn.computeProblemData(t, self.qdes, self.vdes)
		# Get the initial data
		self.data = self.invdyn.data()
		
		# Task definition
		"""self.FRfootTask = tsid.TaskSE3Equality("FR-foot-positioning", self.robot, 'FR_FOOT')
		self.FRfootTask.setKp(kp_foot * matlib.ones(6).T)
		self.FRfootTask.setKd(2.0 * np.sqrt(kp_foot) * matlib.ones(6).T)
		self.FRfootTask.useLocalFrame(False)
		# Add the task to the HQP with weight = w_foot, priority level = 0 (as real constraint) and a transition duration = 0.0
		self.invdyn.addMotionTask(self.FRfootTask, w_foot, 0, 0.0)
		"""
		# POSTURE Task
		self.postureTask = tsid.TaskJointPosture("task-posture", self.robot)
		self.postureTask.setKp(kp_posture * matlib.ones(8).T) # Proportional gain 
		self.postureTask.setKd(2.0 * np.sqrt(kp_posture) * matlib.ones(8).T) # Derivative gain 
		# Add the task to the HQP with weight = 1e-3, priority level = 1 (in the cost function) and a transition duration = 0.0
		self.invdyn.addMotionTask(self.postureTask, w_posture, 0, 0.0)

		
		# TSID Trajectory 
		"""
		pin.forwardKinematics(self.model, self.data, self.qdes)
		pin.updateFramePlacements(self.model, self.data)
	
		FR_foot_ref = self.robot.framePosition(self.data, self.model.getFrameId('FR_FOOT'))
		
		FR_foot_goal = FR_foot_ref.copy()
		FR_foot_goal.translation = np.matrix([FRgoalx, FR_foot_ref.translation[1,0], FRgoalz]).T
		print(FR_foot_ref, FR_foot_goal)
		self.trajFRfoot = tsid.TrajectorySE3Constant("traj_FR_foot", FR_foot_goal)
		"""
		pin.loadReferenceConfigurations(self.model, srdf, False)
		
		q_ref = self.model.referenceConfigurations['straight_standing'] 
		self.trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)
		
		## Initialization of the solver

		# Use EiquadprogFast solver
		self.solver = tsid.SolverHQuadProgFast("qp solver")
		# Resize the solver to fit the number of variables, equality and inequality constraints
		self.solver.resize(self.invdyn.nVar, self.invdyn.nEq, self.invdyn.nIn)

				
	####################################################################
	#                      Torque Control method                       #
	####################################################################
	def control(self, qmes, vmes, t):
		
		# Set the trajectory as reference for the position task
		"""sampleFRfoot = self.trajFRfoot.computeNext()
		self.FRfootTask.setReference(sampleFRfoot)"""
		samplePosture = self.trajPosture.computeNext()
		self.postureTask.setReference(samplePosture)
		
		# Resolution of the HQP problem
		self.HQPData = self.invdyn.computeProblemData(t, qmes, vmes)
		self.sol = self.solver.solve(self.HQPData)
		
		# Torques computation
		tau = self.invdyn.getActuatorForces(self.sol)
		
		# Saturation to limit the maximal torque
		t_max = 2.5
		tau = np.maximum(np.minimum(tau, t_max * np.ones((8,1))), -t_max * np.ones((8,1)))
		"(self.sol.status!=0)"
		self.error = self.error or (self.sol.status!=0) or (qmes[0] < -np.pi/2) or (qmes[2] < -np.pi/2) or (qmes[4] < -np.pi/2) or (qmes[6] < -np.pi/2) or (qmes[0] > np.pi/2) or (qmes[2] > np.pi/2) or (qmes[4] > np.pi/2) or (qmes[6] > np.pi/2)
		if (self.error): print(self.sol.status)
		return tau

# Parameters of the desired trajectory

q0 = np.ones((8,1))		# initial configuration

FRgoalz = 0.1
FRgoalx = 0.1
