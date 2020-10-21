import numpy as np
import time
import pybullet as p
from gym import error, spaces, utils

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary


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
    def __init__(self, drone_model: DroneModel=DroneModel.CF2X, num_drones: int=1, \
                        visibility_radius: float=np.inf, initial_xyzs=None, initial_rpys=None, \
                        physics: Physics=Physics.PYB_DRAG, freq: int=200, aggregate_phy_steps: int=1, \
                        gui=False, record=False, obstacles=False):
        if num_drones!=1: print("[ERROR] in RLTakeoffAviary.__init__(), RLTakeoffAviary only accepts num_drones=1" ); exit()
        super().__init__(drone_model=drone_model, visibility_radius=visibility_radius, \
            initial_xyzs=initial_xyzs, initial_rpys=initial_rpys, physics=physics, freq=freq, aggregate_phy_steps=aggregate_phy_steps, \
            gui=gui, record=record, obstacles=obstacles) 
            
        self.geoFenceMax = 0.99
        
        self.penaltyPosition = 1
        self.penaltyAngle = 1
        self.penaltyVelocity = 5
        self.penaltyAngularVelocity = 5
        self.penaltyFlag = 1000
        #self.penaltyControl = 0.5/250**2
        #self.penaltyDiffControl = 0.5/250**2
        
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
        
        #### Initialize the drones kinemaatic information ##################################################
        unitVector = np.random.rand(3)-np.array([0.5,0.5,0])
        startingPoint = np.ones((self.NUM_DRONES,3))*np.array([0,0,0.5])
        self.pos = 0.5*np.random.rand(1)*unitVector/np.linalg.norm(unitVector)*np.ones((self.NUM_DRONES,3))+startingPoint

        self.quat = np.zeros((self.NUM_DRONES,4)); self.rpy = np.zeros((self.NUM_DRONES,3))
        self.vel = np.zeros((self.NUM_DRONES,3)); self.ang_v = np.zeros((self.NUM_DRONES,3))
        #### Initialize wind speed and heading information #################################################
        windHeading = np.random.rand()*2*np.pi
        self.wind=np.random.rand()*self.MAXWINDSPEED*np.array([-np.cos(windHeading),np.sin(windHeading),0])

        #### Set PyBullet's parameters #####################################################################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #################################################
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        self.DRONE_IDS = np.array([p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+self.URDF, self.INIT_XYZS[i,:], p.getQuaternionFromEuler(self.INIT_RPYS[i,:]), physicsClientId=self.CLIENT) for i in range(self.NUM_DRONES)])
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
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([-1,           -1,           -1,           -1])
        act_upper_bound = np.array([1,            1,            1,            1])
        return spaces.Box( low=act_lower_bound, high=act_upper_bound, dtype=np.float32 )
        
    ####################################################################################################
    #### Return the observation space of the environment, a Box(20,) ###################################
    ####################################################################################################
    def _observationSpace(self):
        #### Observation vector ### X        Y        Z      R       P       Y       VX       VY       VZ       WR       WP       WY       P0            P1            P2            P3
        obs_lower_bound = np.array([-1,      -1,      0,     -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1,      -1,           -1,           -1,           -1])
        obs_upper_bound = np.array([1,       1,       1,     1,      1,      1,      1,       1,       1,       1,       1,       1,       1,            1,            1,            1])
        return spaces.Box( low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32 )

    ####################################################################################################
    #### Return the current observation of the environment #############################################
    ####################################################################################################
    #### Returns #######################################################################################
    #### - obs (20,) array                  for its content see _observationSpace() ####################
    ####################################################################################################
    def _computeObs(self):
        droneState = self._getDroneState(0)
        droneState = np.hstack([droneState[0:3],droneState[7:20]])
        droneState.reshape(16,)        
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
        rpm = self._normActionToRPM(action)
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
        xy = obs[0:2]
        z = obs[2]
        v = obs[6:9]
        angle = obs[3:6]
        omega = obs[9:12]
        #actions = obs[12:16]
        
        errorPosition = np.sum(np.square(xy))+np.square(z-0.5)
        errorVelocity = np.sum(np.square(v))
        errorAngularVelocity = np.sum(np.square(omega))
        
        penaltyPosition = errorPosition*self.penaltyPosition
        penaltyAngle = np.square(angle[2])*self.penaltyAngle
        penaltyVelocity = errorVelocity*self.penaltyVelocity
        penaltyAngularVelocity = errorAngularVelocity*self.penaltyAngularVelocity
        
        outOfGeoFence = any(np.abs(obs[0:3]) > self.geoFenceMax)
        crashed = True if obs[2]<self.COLLISION_H else False
        penaltyFlag = self.penaltyFlag if outOfGeoFence or crashed else 0
        
        rewardPosition = np.sqrt(errorPosition)>self.positionThreshold
        rewardYaw = angle[2]>self.angleThreshold
        rewardVelocity = np.sqrt(errorVelocity)>self.velocityThreshold
        rewardAngularVelocity = np.sqrt(errorAngularVelocity)>self.angularVelocityThreshold
        
        if all([rewardPosition, rewardYaw, rewardVelocity, rewardAngularVelocity]):
            rewardGoal = self.rewardGoal
        else:
            rewardGoal = 0
            
        #actions = np.abs(actions)
        #penaltyControl = np.sum(np.square(actions-self.HOVER_RPM))*self.penaltyControl
        #penaltyDiffControl = np.square(np.max(actions)-np.min(actions))*self.penaltyDiffControl
        
        return rewardGoal - penaltyPosition - penaltyAngle - penaltyVelocity - penaltyAngularVelocity
        -penaltyFlag
        #- penaltyControl - penaltyDiffControl

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
        return np.hstack([normalized_pos, normalized_rp, normalized_y, normalized_vel, normalized_ang_vel_rp, normalized_ang_vel_y, state[12:16] ]).reshape(16,)

    ####################################################################################################
    #### Print a warning if any of the 20 values in a state vector is out of the normalization range ###
    ####################################################################################################
    def _clipAndNormalizeStateWarning(self, state, clipped_pos, clipped_rp, clipped_vel, clipped_ang_vel_rp, clipped_ang_vel_y):
        if not(clipped_pos==np.array(state[0:3])).all(): print("[WARNING] it", self.step_counter, "in RLTakeoffAviary._clipAndNormalizeState(), out-of-bound position [{:.2f} {:.2f} {:.2f}], consider a more conservative implementation of RLTakeoffAviary._computeDone()".format(state[0], state[1], state[2]))
        if not(clipped_rp==np.array(state[3:5])).all(): print("[WARNING] it", self.step_counter, "in RLTakeoffAviary._clipAndNormalizeState(), out-of-bound roll/pitch [{:.2f} {:.2f}], consider a more conservative implementation of RLTakeoffAviary._computeDone()".format(state[3], state[4]))
        if not(clipped_vel==np.array(state[6:9])).all(): print("[WARNING] it", self.step_counter, "in RLTakeoffAviary._clipAndNormalizeState(), out-of-bound velocity [{:.2f} {:.2f} {:.2f}], consider a more conservative implementation of RLTakeoffAviary._computeDone()".format(state[6], state[7], state[8]))
        if not(clipped_ang_vel_rp==np.array(state[9:11])).all(): print("[WARNING] it", self.step_counter, "in RLTakeoffAviary._clipAndNormalizeState(), out-of-bound angular velocity [{:.2f} {:.2f} {:.2f}], consider a more conservative implementation of RLTakeoffAviary._computeDone()".format(state[9], state[10], state[11]))
        if not(clipped_ang_vel_y==np.array(state[11])): print("[WARNING] it", self.step_counter, "in RLTakeoffAviary._clipAndNormalizeState(), out-of-bound angular velocity [{:.2f} {:.2f} {:.2f}], consider a more conservative implementation of RLTakeoffAviary._computeDone()".format(state[9], state[10], state[11]))