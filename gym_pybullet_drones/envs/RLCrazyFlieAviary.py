import numpy as np
import time
import pybullet as p
import pybullet_data
import os
from scipy.optimize import nnls
from gym import error, spaces, utils

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.utils.utils import *


######################################################################################################################################################
#### Single drone environment class for reinforcement learning applications (in this implementation, taking off from the origin) #####################
######################################################################################################################################################
class RLCrazyFlieAviary(BaseAviary):

    ####################################################################################################
    #### Initialize the environment ####################################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - drone_model (DroneModel)         desired drone type (associated to an .urdf file) ###########
    #### - num_drones (int)                 desired number of drones in the aviary #####################
    #### - visibility_radius (float)        used to compute the drones' adjacency matrix, in meters ####
    #### - initial_xyzs ((3,1) array)       initial XYZ position of the drones #########################
    #### - initial_rpys ((3,1) array)       initial orientations of the drones (radians) ###############
    #### - physics (Physics)                desired implementation of physics/dynamics #################
    #### - freq (int)                       the frequency (Hz) at which the physics engine advances ####
    #### - aggregate_phy_steps (int)        number of physics updates within one call of .step() #######
    #### - gui (bool)                       whether to use PyBullet's GUI ##############################
    #### - record (bool)                    whether to save a video of the simulation ##################
    #### - obstacles (bool)                 whether to add obstacles to the simulation #################
    ####################################################################################################
    def __init__(self, drone_model: DroneModel=DroneModel.CF2X,
                        neighbourhood_radius: float=np.inf, initial_xyzs=None, initial_rpys=None,
                        physics: Physics=Physics.PYB_DRAG, freq: int=200, aggregate_phy_steps: int=1,
                        gui=False, record=False, obstacles=False, maxWindSpeed=8.0, user_debug_gui=True,
                        PID_Control=False, target_pos=None, run_name: str=''):

        self.usePID = PID_Control
        
        super().__init__(drone_model=drone_model, neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs, initial_rpys=initial_rpys, physics=physics, freq=freq, aggregate_phy_steps=aggregate_phy_steps,
            gui=gui, record=record, obstacles=obstacles, maxWindSpeed=maxWindSpeed, user_debug_gui=user_debug_gui, run_name=run_name) 
        
        if target_pos is None:
            self.target_pos = np.array([0,0,0.5])
        elif np.array(target_pos).shape==(1,3):
            self.target_pos=target_pos
        else:
            prRed("[ERRROR] invalid target position shape in RLCrazyFlieAviary.__init__()")

        self.A = np.array([ [1, 1, 1, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [-1, 1, -1, 1] ]); self.INV_A = np.linalg.inv(self.A)
        self.B_COEFF = np.array([1/self.KF, 1/(self.KF*self.L), 1/(self.KF*self.L), 1/self.KM])
        self.MAX_ROLL_PITCH = np.pi/6

        self.geoFenceMax = 0.99
        
        self.penaltyPosition = 1
        self.penaltyAngle = 1
        self.penaltyVelocity = 5
        self.penaltyAngularVelocity = 5
        self.penaltyFlag = 1000
        
        self.positionThreshold = 0.1
        self.angleThreshold = np.pi/18
        self.angularVelocityThreshold = 0.05
        self.velocityThreshold = 0.1
        
        self.rewardGoal = 20
        
    def _housekeeping(self):
        #### Initialize/reset counters and zero-valued variables ###########################################
        self.RESET_TIME = time.time(); self.step_counter = 0; self.first_render_call = True
        self.X_AX = -1*np.ones(self.NUM_DRONES); self.Y_AX = -1*np.ones(self.NUM_DRONES); self.Z_AX = -1*np.ones(self.NUM_DRONES);
        self.GUI_INPUT_TEXT = -1*np.ones(self.NUM_DRONES); self.USE_GUI_RPM=False; self.last_input_switch = 0
        self.last_action = -1*np.ones((self.NUM_DRONES,4))
        self.last_clipped_action = np.zeros((self.NUM_DRONES,4)); self.gui_input = np.zeros(4)
        self.no_pybullet_dyn_accs = np.zeros((self.NUM_DRONES,3))
        
        #### Initialize the drones kinematic information ###################################################
        unitVector = np.random.rand(3)-np.array([0.5,0.5,0])
        startingPoint = np.ones((self.NUM_DRONES,3))*np.array([0,0,0.5])
        self.pos = 0.5*np.random.rand(1)*unitVector/np.linalg.norm(unitVector)*np.ones((self.NUM_DRONES,3))+startingPoint

        self.quat = np.zeros((self.NUM_DRONES,4)); self.rpy = np.zeros((self.NUM_DRONES,3))
        self.vel = np.zeros((self.NUM_DRONES,3)); self.ang_v = np.zeros((self.NUM_DRONES,3))

        #### Initialize wind speed and heading information #################################################
        windHeading = np.random.rand()*2*np.pi
        self.wind=np.random.rand()*self.MAXWINDSPEED*np.array([-np.cos(windHeading),np.sin(windHeading),0])
        
        #### Reset Controller information ##################################################################
        self.last_pos_e = np.zeros(3); self.integral_pos_e = np.zeros(3); self.last_rpy_e = np.zeros(3); self.integral_rpy_e = np.zeros(3)

        #### Set PyBullet's parameters #####################################################################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #################################################
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        self.DRONE_IDS = np.array([p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+self.URDF, self.pos[i,:], p.getQuaternionFromEuler(self.rpy[i,:]), physicsClientId=self.CLIENT) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            #### Show the frame of reference of the drone, note thet it can severly slow down the GUI ##########
            if self.GUI and self.USER_DEBUG: self._showDroneLocalAxes(i)
            #### Disable collisions between drones' and the ground plane, e.g., to start a drone at [0,0,0] ####
            p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.OBSTACLES: self._addObstacles()


    ####################################################################################################
    #### Return the action space of the environment, a Box(4,) #########################################
    ####################################################################################################
    def _actionSpace(self):
        if self.usePID:
            #### PID Gain vector ###### KPx   KPy   KPz   KIx   KIy   Kiz   KDx   KDy   KDz   KPr   KPp   KPy   KIr   KIp   KIy   KDr   KDp   KDy
            act_lower_bound = np.array([ 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0])
            act_upper_bound = np.array([ 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1])
        else:
            #### Action vector ######## P0            P1            P2            P3
            act_lower_bound = np.array([-1,           -1,           -1,           -1])
            act_upper_bound = np.array([1,            1,            1,            1])
        return spaces.Box( low=act_lower_bound, high=act_upper_bound, dtype=np.float32 )
        
    ####################################################################################################
    #### Return the observation space of the environment, a Box(20,) ###################################
    ####################################################################################################
    def _observationSpace(self):
        #### Observation vector ### X        Y        Z      R       P       Y       VX       VY       VZ       WR       WP       WY
        obs_lower_bound = np.array([-1,      -1,      0,     -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1])
        obs_upper_bound = np.array([1,       1,       1,     1,      1,      1,      1,       1,       1,       1,       1,       1])
        return spaces.Box( low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32 )

    ####################################################################################################
    #### Return the current observation of the environment #############################################
    ####################################################################################################
    #### Returns #######################################################################################
    #### - obs (20,) array                  for its content see _observationSpace() ####################
    ####################################################################################################
    def _computeObs(self):
        droneState = self._getDroneStateVector(0)
        droneState = np.hstack([droneState[0:3],droneState[7:16]])
        droneState.reshape(np.shape(self.observation_space)[0],)        
        return self._clipAndNormalizeState(droneState)

    ####################################################################################################
    #### Preprocess the action passed to step() ########################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - action ((4,1) array)             unclipped RPMs commanded to the 4 motors of the drone ######
    ####################################################################################################
    #### Returns #######################################################################################
    #### - clipped_action ((4,1) array)     clipped RPMs commanded to the 4 motors of the drone ########
    ####################################################################################################
    def _preprocessAction(self, action):
        if self.usePID:
            rpm = self._caluclatePIDControlSignal(action)
        else:
            rpm = self._normalizedActionToRPM(action)
        return np.clip(np.array(rpm), 0, self.MAX_RPM)

    ####################################################################################################
    #### Compute the current reward value(s) ###########################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - obs (..)                         the return of _computeObs() ################################
    ####################################################################################################
    #### Returns #######################################################################################
    #### - reward (..)                      the reward(s) associated to the current obs/state ##########
    ####################################################################################################
    def _computeReward(self, obs):
        pos = obs[0:3]
        v = obs[6:9]
        angle = obs[3:6]
        omega = obs[9:12]
        
        errorPosition = np.sum(np.square(pos-self.target_pos))
        errorVelocity = np.sum(np.square(v))
        errorAngularVelocity = np.sum(np.square(omega))
        
        penaltyPosition = errorPosition*self.penaltyPosition
        penaltyAngle = np.square(angle[2])*self.penaltyAngle
        penaltyVelocity = errorVelocity*self.penaltyVelocity
        penaltyAngularVelocity = errorAngularVelocity*self.penaltyAngularVelocity
        
        outOfGeoFence = any(np.abs(pos) > self.geoFenceMax)
        crashed = True if pos[2]<self.COLLISION_H else False
        penaltyFlag = self.penaltyFlag if outOfGeoFence or crashed else 0
        
        rewardPosition = np.sqrt(errorPosition)>self.positionThreshold
        rewardYaw = angle[2]>self.angleThreshold
        rewardVelocity = np.sqrt(errorVelocity)>self.velocityThreshold
        rewardAngularVelocity = np.sqrt(errorAngularVelocity)>self.angularVelocityThreshold
        
        if all([rewardPosition, rewardYaw, rewardVelocity, rewardAngularVelocity]):
            rewardGoal = self.rewardGoal
        else:
            rewardGoal = 0
        
        #return rewardGoal - penaltyPosition - penaltyAngle - penaltyVelocity - penaltyAngularVelocity
        #-penaltyFlag
        return -penaltyPosition - penaltyAngle

    ####################################################################################################
    #### Compute the current done value(s) #############################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - obs (..)                         the return of _computeObs() ################################
    ####################################################################################################
    #### Returns #######################################################################################
    #### - done (..)                        the done value(s) associated to the current obs/state ######
    ####################################################################################################
    def _computeDone(self, obs):
        outOfGeoFence = any(np.abs(obs[0:3]) > self.geoFenceMax)
        outOfTime = True if (self.step_counter/self.SIM_FREQ > 10) else False
        crashed = True if obs[2]<self.COLLISION_H else False
        return outOfGeoFence or outOfTime or crashed 

    ####################################################################################################
    #### Compute the current info dict(s) ##############################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - obs (..)                         the return of _computeObs() ################################
    ####################################################################################################
    #### Returns #######################################################################################
    #### - info (..)                        the info dict(s) associated to the current obs/state #######
    ####################################################################################################
    def _computeInfo(self, obs):
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    ####################################################################################################
    #### Normalize the 20 values in the simulation state to the [-1,1] range ###########################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - state ((20,1) array)             raw simulation state #######################################
    ####################################################################################################
    #### Returns #######################################################################################
    #### - normalized state ((20,1) array)  clipped and normalized simulation state ####################
    ####################################################################################################
    def _clipAndNormalizeState(self, state):
        clipped_pos = np.clip(state[0:3], -1, 1)
        clipped_rp = np.clip(state[3:5], -np.pi/3, np.pi/3)
        clipped_vel = np.clip(state[6:9], -5, 5)
        clipped_ang_vel_rp = np.clip(state[9:11], -10*np.pi, 10*np.pi)
        clipped_ang_vel_y = np.clip(state[11], -20*np.pi, 20*np.pi)
        if self.GUI: self._clipAndNormalizeStateWarning(state, clipped_pos, clipped_rp, clipped_vel, clipped_ang_vel_rp, clipped_ang_vel_y)
        normalized_pos = clipped_pos
        normalized_rp = clipped_rp/(np.pi/3)
        normalized_y = state[5]/np.pi
        normalized_vel = clipped_vel/5
        normalized_ang_vel_rp = clipped_ang_vel_rp/(10*np.pi)
        normalized_ang_vel_y = clipped_ang_vel_y/(20*np.pi)
        return np.hstack([normalized_pos, normalized_rp, normalized_y, normalized_vel, normalized_ang_vel_rp, normalized_ang_vel_y, state[12:16] ]).reshape(np.shape(self.observation_space)[0],)

    ####################################################################################################
    #### Print a warning if any of the 20 values in a state vector is out of the normalization range ###
    ####################################################################################################
    def _clipAndNormalizeStateWarning(self, state, clipped_pos, clipped_rp, clipped_vel, clipped_ang_vel_rp, clipped_ang_vel_y):
        if not(clipped_pos==np.array(state[0:3])).all(): prYellow("[WARNING] at {} in RLCrazyFlieAviary._clipAndNormalizeState(), out-of-bound position [{:.2f} {:.2f} {:.2f}], consider a more conservative implementation of RLCrazyFlieAviary._computeDone()".format(self.step_counter,state[0], state[1], state[2]))
        if not(clipped_rp==np.array(state[3:5])).all(): prYellow("[WARNING] at {} in RLCrazyFlieAviary._clipAndNormalizeState(), out-of-bound roll/pitch [{:.2f} {:.2f}], consider a more conservative implementation of RLCrazyFlieAviary._computeDone()".format(self.step_counter,state[3], state[4]))
        if not(clipped_vel==np.array(state[6:9])).all(): prYellow("[WARNING] at {} in RLCrazyFlieAviary._clipAndNormalizeState(), out-of-bound velocity [{:.2f} {:.2f} {:.2f}], consider a more conservative implementation of RLCrazyFlieAviary._computeDone()".format(self.step_counter,state[6], state[7], state[8]))
        if not(clipped_ang_vel_rp==np.array(state[9:11])).all(): prYellow("[WARNING] at {} in RLCrazyFlieAviary._clipAndNormalizeState(), out-of-bound angular velocity [{:.2f} {:.2f} {:.2f}], consider a more conservative implementation of RLCrazyFlieAviary._computeDone()".format(self.step_counter,state[9], state[10], state[11]))
        if not(clipped_ang_vel_y==np.array(state[11])): prYellow("[WARNING] at {} in RLCrazyFlieAviary._clipAndNormalizeState(), out-of-bound angular velocity [{:.2f} {:.2f} {:.2f}], consider a more conservative implementation of RLCrazyFlieAviary._computeDone()".format(self.step_counter,state[9], state[10], state[11]))

    ####################################################################################################
    #### Compute the control action for a single drone #################################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - gains ((18,1) array)              PID Gains #################################################
    ####################################################################################################
    #### Returns #######################################################################################
    #### - rpm ((4,1) array)                RPM values to apply to the 4 motors ########################
    ####################################################################################################
    def _caluclatePIDControlSignal(self, gains):

        droneState = self._getDroneStateVector(0)
        cur_pos = droneState[0:3]
        cur_quat = droneState[3:7]
        
        Kpos = gains[0:9]
        Katt = gains[9:18]
        
        thrust, computed_target_rpy = self._simplePIDPositionControl(cur_pos, cur_quat, Kpos)
        rpm = self._simplePIDAttitudeControl(thrust, cur_quat, computed_target_rpy, Katt)
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm
        
    ####################################################################################################
    #### Generic PID position control (with yaw locked to 0.) ##########################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - cur_pos ((3,1) array)            current position ###########################################
    #### - cur_quat ((4,1) array)           current orientation as a quaternion ########################
    #### - K ((9,1) array)                  PID gains for the position control #########################
    ####################################################################################################
    #### Returns #######################################################################################
    #### - thrust (float)                   thrust along the drone z-axis ##############################
    #### - target_rpy ((3,1) array)         computed target roll, pitch, and yaw #######################
    #### - yaw_e (float)                    current yaw error ##########################################
    ####################################################################################################
    def _simplePIDPositionControl(self, cur_pos, cur_quat, K):
        pos_e = self.target_pos - np.array(cur_pos).reshape(3)
        d_pos_e = (pos_e - self.last_pos_e) * self.SIM_FREQ
        self.last_pos_e = pos_e
        self.integral_pos_e = self.integral_pos_e + pos_e/self.SIM_FREQ
        #### PID target thrust #############################################################################
        target_force = np.array([0,0,self.GRAVITY]) + np.multiply(K[0:3]/2,pos_e) + np.multiply(K[3:6]/1000,self.integral_pos_e) + np.multiply(K[6:9]/2,d_pos_e)
        target_rpy = np.zeros(3)
        sign_z =  np.sign(target_force[2])
        if sign_z==0: sign_z = 1
        #### Target rotation ###############################################################################
        target_rpy[0] = np.arcsin(-sign_z*target_force[1] / np.linalg.norm(target_force))
        target_rpy[1] = np.arctan2(sign_z*target_force[0], sign_z*target_force[2])
        target_rpy[2] = 0.
        target_rpy[0] = np.clip(target_rpy[0], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)
        target_rpy[1] = np.clip(target_rpy[1], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3,3)
        thrust = np.dot(cur_rotation, target_force)
        return thrust[2], target_rpy

    ####################################################################################################
    #### Generic PID attitude control (with yaw locked to 0.) ##########################################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - thrust (float)                   desired thrust along the drone z-axis ######################
    #### - cur_quat ((4,1) array)           current orientation as a quaternion ########################
    #### - target_rpy ((3,1) array)         computed target roll, pitch, and yaw #######################
    #### - K ((9,1) array)                  PID gains for the attitude control #########################
    ####################################################################################################
    #### Returns #######################################################################################
    #### - rpm ((4,1) array)                RPM values to apply to the 4 motors ########################
    ####################################################################################################
    def _simplePIDAttitudeControl(self, thrust, cur_quat, target_rpy, K):
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        rpy_e = target_rpy - np.array(cur_rpy).reshape(3,)
        if rpy_e[2]>np.pi: rpy_e[2] = rpy_e[2] - 2*np.pi
        if rpy_e[2]<-np.pi: rpy_e[2] = rpy_e[2] + 2*np.pi
        d_rpy_e = (rpy_e - self.last_rpy_e) * self.SIM_FREQ
        self.last_rpy_e = rpy_e
        self.integral_rpy_e = self.integral_rpy_e + rpy_e/self.SIM_FREQ
        #### PID target torques ############################################################################
        target_torques = np.multiply(K[0:3]/2,rpy_e) + np.multiply(K[3:6]/1000,self.integral_rpy_e) + np.multiply(K[6:9]/2,d_rpy_e)
        return self._nnlsRPM(thrust, target_torques[0], target_torques[1], target_torques[2])

    ####################################################################################################
    #### Non-negative Least Squares (NNLS) RPM from desired thrust and torques  ########################
    ####################################################################################################
    #### Arguments #####################################################################################
    #### - thrust (float)                   desired thrust along the local z-axis ######################
    #### - x_torque (float)                 desired x-axis torque ######################################
    #### - y_torque (float)                 desired y-axis torque ######################################
    #### - z_torque (float)                 desired z-axis torque ######################################
    ####################################################################################################
    #### Returns #######################################################################################
    #### - rpm ((4,1) array)                RPM values to apply to the 4 motors ########################
    ####################################################################################################
    def _nnlsRPM(self, thrust, x_torque, y_torque, z_torque):
        new_line = True
        #### Check the feasibility of thrust and torques ###################################################
        if thrust<0 or thrust>self.MAX_THRUST:
            if new_line: print(); new_line = False
            prYellow("[WARNING] ctrl at {} in RLCrazyFlieAviary._nnlsRPM(), unfeasible thrust {:.2f} outside range [0, {:.2f}]".format(self.step_counter, thrust, self.MAX_THRUST))
        if np.abs(x_torque)>self.MAX_XY_TORQUE:
            if new_line: print(); new_line = False
            prYellow("[WARNING] ctrl at {} in RLCrazyFlieAviary._nnlsRPM(), unfeasible roll torque {:.2f} outside range [{:.2f}, {:.2f}]".format(self.step_counter, x_torque, -self.MAX_XY_TORQUE, self.MAX_XY_TORQUE))
        if np.abs(y_torque)>self.MAX_XY_TORQUE:
            if new_line: print(); new_line = False
            prYellow("[WARNING] ctrl at {} in RLCrazyFlieAviary._nnlsRPM(), unfeasible pitch torque {:.2f} outside range [{:.2f}, {:.2f}]".format(self.step_counter, y_torque, -self.MAX_XY_TORQUE, self.MAX_XY_TORQUE))
        if np.abs(z_torque)>self.MAX_Z_TORQUE:
            if new_line: print(); new_line = False
            prYellow("[WARNING] ctrl at {} in RLCrazyFlieAviary._nnlsRPM(), unfeasible yaw torque {:.2f} outside range [{:.2f}, {:.2f}]".format(self.step_counter, z_torque, -self.MAX_Z_TORQUE, self.MAX_Z_TORQUE))
        B = np.multiply(np.array([thrust, x_torque, y_torque, z_torque]), self.B_COEFF)
        sq_rpm = np.dot(self.INV_A, B)
        #### Use NNLS if any of the desired angular velocities is negative #################################
        if np.min(sq_rpm)<0:
            sol, res = nnls(self.A, B, maxiter=3*self.A.shape[1])
            if new_line: print(); new_line = False
            prYellow("[WARNING] ctrl at {} in RLCrazyFlieAviary._nnlsRPM(), unfeasible squared rotor speeds, using NNLS".format(self.step_counter))
            print("Negative sq. rotor speeds:\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(sq_rpm[0], sq_rpm[1], sq_rpm[2], sq_rpm[3]),
                    "\t\tNormalized: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(sq_rpm[0]/np.linalg.norm(sq_rpm), sq_rpm[1]/np.linalg.norm(sq_rpm), sq_rpm[2]/np.linalg.norm(sq_rpm), sq_rpm[3]/np.linalg.norm(sq_rpm)))
            print("NNLS:\t\t\t\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(sol[0], sol[1], sol[2], sol[3]),
                    "\t\t\tNormalized: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(sol[0]/np.linalg.norm(sol), sol[1]/np.linalg.norm(sol), sol[2]/np.linalg.norm(sol), sol[3]/np.linalg.norm(sol)),
                    "\t\tResidual: {:.2f}".format(res) )
            sq_rpm = sol
        return np.sqrt(sq_rpm)