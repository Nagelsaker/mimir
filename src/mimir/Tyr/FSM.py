import imp
import json
import csv
import time
import rospy
import gym
import os

import numpy as np

from datetime import date
from PyQt5.QtGui import QImage
from mimir.Tyr.Utility.utils import drawLandmarks, loadWorkspace, generateFilename
from mimir.Tyr.Comms.Controller import Controller, Obstacle
from mimir.Tyr.Hand.HandTracking import HandTracking
from mimir.Tyr.Hand.HandModel import *
from mimir.msg import LeverPose
from mimir.DDPG.ddpg_agent import ddpg_agent
from mimir.DDPG import arguments
from mimir.DDPG.train import get_env_params
from open_manipulator_rl_environments.task_environments.lever_pull_task import OpenManipulatorLeverPullEnvironment



class FSM():
    '''
    A class representing a finite state machine

    The FSM swicthes between different control states for the Robotis
    OpenManipulator, based on the input signals from an operator.
    For further information on the inpu signals, see the documentation
    on HandModel and HandTracking.

    Attributes
    ----------
        depthRange : 2x1 Array(Float)
            the range from min to max, which yields max to min manipulator movement
        pathTime: Float
            time it takes for the manipulator to move to desired position
        imgWidth: int
            image width
        imgHeight: int
            image height
        camSN: str
            serial number for the camera recording operators hand
        Kp_default: (5x1) Array(Float)
            controller parameters for the velocity controllers
        handTracker: HandTracking
            hand tracking object
        hm: HandModel
            hand model object
        controller: Controller
            controller object
        obstacles: (5x1) Array(Obstacle)
        imgLM: Array
            current workspace image with hand landmarks drawn
    '''
    STATE = ST_STOP

    def __init__(self):
        rospy.loginfo("TEST")
        # Long range: [0.60, 0.85], Short range: [0.30, 0.59]
        self.depthRange = rospy.get_param("/mimir/depth_range")
        self.pathTime = rospy.get_param("/mimir/path_time") # 0.2
        self.imgWidth = rospy.get_param("/mimir/img_width")
        self.imgHeight = rospy.get_param("/mimir/img_height")
        self.camSN = rospy.get_param("/mimir/sn_cam")
        useDepth = rospy.get_param("/mimir/use_depth")
        # Kp=[K_p_beta, K_p_r, K_p_z, K_p_theta, K_p_phi]
        self.Kp_default = rospy.get_param("/mimir/Kp_default")
        wristAngle_threshold = rospy.get_param("/mimir/threshold_angle_wrist")
        thumbAngle_threshold = rospy.get_param("/mimir/threshold_angle_thumb")
        fingerAngle_threshold = rospy.get_param("/mimir/threshold_angle_finger")

        # Obstacles
        floor = Obstacle(zRange=rospy.get_param("/mimir/ob_floor/zRange"))
        ceiling = Obstacle(zRange=rospy.get_param("/mimir/ob_ceiling/zRange"))
        innerCylinder = Obstacle(radiusRange=rospy.get_param("/mimir/ob_inner_cylinder/radiusRange"))
        outerCylinder = Obstacle(radiusRange=rospy.get_param("/mimir/ob_outer_cylinder/radiusRange"))
        motor = Obstacle(xRange=rospy.get_param("/mimir/ob_motor/xRange"),
                         yRange=rospy.get_param("/mimir/ob_motor/yRange"),
                         zRange=rospy.get_param("/mimir/ob_motor/zRange"))
        self.obstacles = np.array([floor, ceiling, innerCylinder, outerCylinder, motor])

        # Operator workspace
        self.workspaceOverlay, self.workspaceSections = loadWorkspace()

        self.handTracker = HandTracking(self.camSN)
        self.hm = HandModel("left", self.workspaceSections, wristAngle_threshold, thumbAngle_threshold, fingerAngle_threshold, useDepth)
        self.controller = Controller(self.imgWidth, self.imgHeight, Kp=self.Kp_default, pathTime=self.pathTime, obstacles=self.obstacles)
        self.imgLM = None

        # RL Agent
        args = rospy.get_param("/mimir/DDPG")
        env = gym.make(args["env_name"])
        env_params = get_env_params(env)
        model_path = args["load_model_path"]
        load_model_path = f"{os.path.dirname(os.path.abspath(__file__))}/../DDPG/{model_path}"
        self.agent = ddpg_agent(args, env, env_params, load_model_path)
        self.agent.setGoal()
        self.currentGoal = self.agent.getCurrentGoal()

        self.goalReachedTime = -10.0
        self.leverIconOnTime = -10.0
        self.goalReached = False
        self.goalThresholdTime = 1.0

        # Setup logging
        self.leverPoseSub = rospy.Subscriber("/mimir/lever_angle_pose", LeverPose, self._leverPoseCallback)
        self.leverPose = LeverPose()

        self.pathToLog = rospy.get_param("/mimir/path_to_log")
        self.writeLogs = rospy.get_param("/mimir/write_logs")
        self.pathToLogPose = ""
        self.pathToLogFSMState = ""
        self.pathToLogHandPoints = ""
        self.pathToLogTimeSteps = ""
        self.pathToLogLeverPose = ""
        self.pathToLogRewardSuccess = ""
        self.curDate = date.today().strftime("%Y_%m_%d")
        self.curTime = 0
        self.startTime = time.time()
        if self.writeLogs:
            self.pathToLogPose = generateFilename(self.pathToLog, "csv", f"{self.curDate}_RobotPoseLogger")
            self.pathToLogFSMState = generateFilename(self.pathToLog, "csv", f"{self.curDate}_FSMStateLogger")
            self.pathToLogHandPoints = generateFilename(self.pathToLog, "csv", f"{self.curDate}_HandPointsLogger")
            self.pathToLogTimeSteps = generateFilename(self.pathToLog, "csv", f"{self.curDate}_TimeStepLogger")
            self.pathToLogLeverPose = generateFilename(self.pathToLog, "csv", f"{self.curDate}_LeverPoseLogger")
            self.pathToLogRewardSuccess = generateFilename(self.pathToLog, "csv", f"{self.curDate}_RewardSuccessLogger")


    def run(self, thread=None):
        '''
        In:
            thread: (QThread) object
        '''
        self.controller.updateRobotPose(updateX=True, updateY=True, updateZ=True)
        self.handTracker.startStream()

        try:
            while True: # Tracking loop
                handPoints, image, results = self.handTracker.getLiveLandamarks()

                if thread is not None:
                    if thread.isInterruptionRequested():
                        raise Exception

                    self.imgLM =  drawLandmarks(results, image, self.workspaceOverlay, thread)
                    h, w, ch = self.imgLM.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(self.imgLM.data, w, h, bytesPerLine, QImage.Format_RGB888).scaled(thread.w, thread.h)
                    thread.changePixmap.emit(convertToQtFormat)
                
                self.hm.addMeasurement(handPoints)
                depth,_ = self.hm.getHandDepth()
                
                self.controller.updateRobotPose()
                currentGesture = self.hm.getCurrentGesture()
                wsLoc = self.hm.getWorkspaceLocation(self.imgHeight, self.imgWidth)

                if len(handPoints) != 0 and thread is not None:
                    # Send Current gesture to GUI application
                    thread.activateGesture.emit(currentGesture)
                    thread.updateSkeleton.emit(handPoints[0])
                    thread.setDepthValue.emit(depth*100)

                if self.leverPose.measured_position != [] and thread is not None:
                    thread.setMeasuredLeverAngle.emit(np.rad2deg(self.leverPose.measured_angle))
                    thread.setMeasuredLeverPosition.emit(self.leverPose.measured_position[0])
                    thread.setEstLeverAngle.emit(np.rad2deg(self.leverPose.estimated_angle))
                    thread.setEstLeverPos.emit(self.leverPose.estimated_position[0])

                    # Toggle lever icon
                    # if self.goalReached:
                    #     if self.goalReachedTime == -10.0:
                    #         self.goalReachedTime = time.time()
                    #     if time.time() - self.goalReachedTime > self.goalThresholdTime:
                    #         if self.leverIconOnTime == -10.0:
                    #             self.leverIconOnTime = time.time()
                    #         if time.time() - self.leverIconOnTime < 3.0: # Light the lever icon for 3 seconds
                    #             thread.setLeverStatusIcon(True)
                    #         else:
                    #             self.leverIconOnTime = -10.0
                    #             thread.setLeverStatusIcon(False)
                    #             self.goalReached = False
                    #             self.goalReachedTime = -10.0
                
                    if self.goalReached:
                        self.leverIconOnTime = time.time()

                    if time.time() - self.leverIconOnTime < 3.0: # Light the lever icon for 3 seconds
                        thread.setLeverStatusIcon.emit(True)
                    else:
                        self.leverIconOnTime = -10.0
                        thread.setLeverStatusIcon.emit(False)
                        self.goalReached = False
            

                # Current time
                self.curTime = time.time() - self.startTime
                # rospy.loginfo(f"Status: {currentGesture}, \tTime: {self.curTime}")

                # FSM
                usePrecision = (currentGesture==PRECISION)
                if wsLoc == WS_TURN_LEFT and currentGesture != STOP:
                    self.STATE = ST_TURN_LEFT
                    self.controller.updateRobotPose(updateX=True, updateY=True)
                    self.controller.turnHorizontally(direction="left", precision=usePrecision)
                elif wsLoc == WS_TURN_RIGHT and currentGesture != STOP:
                    self.STATE = ST_TURN_RIGHT
                    self.controller.updateRobotPose(updateX=True, updateY=True)
                    self.controller.turnHorizontally(direction="right", precision=usePrecision)
                elif wsLoc == WS_MOVE_FORWARD and currentGesture != STOP:
                    self.STATE = ST_MOVE_FORWARD
                    self.controller.updateRobotPose(updateX=True, updateY=True)
                    self.controller.incrementRadius(direction="forward", precision=usePrecision)
                elif wsLoc == WS_MOVE_BACKWARD and currentGesture != STOP:
                    self.STATE = ST_MOVE_BACKWARD
                    self.controller.updateRobotPose(updateX=True, updateY=True)
                    self.controller.incrementRadius(direction="backward", precision=usePrecision)
                elif wsLoc == WS_MISC and currentGesture == GRIP:
                    self.STATE = ST_GRIP
                    self.controller.incrementGripper(direction="close")
                elif wsLoc == WS_MISC and currentGesture == UNGRIP:
                    self.STATE = ST_UNGRIP
                    self.controller.incrementGripper(direction="open")
                elif wsLoc == WS_MISC and currentGesture == MOVE_HEIGHT:
                    self.STATE = ST_HEIGHT
                    self.controller.updateRobotPose(updateZ=True)
                    self.controller.incrementHeight(depth=self.hm.getHandDepthSensor(), range=self.depthRange)
                elif wsLoc == WS_MISC and currentGesture == TILT_UP:
                    self.STATE = ST_TILT_UP
                    self.controller.incrementOrientation(direction="up")
                elif wsLoc == WS_MISC and currentGesture == TILT_DOWN:
                    self.STATE = ST_TILT_DOWN
                    self.controller.incrementOrientation(direction="down")
                elif currentGesture == FLIP_HAND:
                    self.STATE = ST_RL_AGENT
                    
                    if np.abs(self.controller.pose["position"]["y"]) < 0.005:
                        self.controller.updateRobotPose(updateX=True, updateY=True, updateZ=True)
                        # rospy.loginfo("STEPPING WITH AI")
                        # Activate RL Agent
                        time.sleep(1e-2)
                        reward, success = self.agent.step()
                        # rospy.loginfo(f"STEPPING WITH AI DONE\t {reward}\t {success}")
                        self.goalReached = success
                        if self.writeLogs:
                            self.logRewardAndSuccess(self.curTime, reward, success, self.currentGoal)

                        if self.goalReached:
                            # Reset the robot to init pose
                            # Find a new goal
                            self.agent.env.reset()
                            self.agent.setGoal()
                            self.currentGoal = self.agent.getCurrentGoal()
                    else:
                        # Turn horizontally
                        # rospy.loginfo("TURNING HORIZONTALLY")
                        self.controller.updateRobotPose(updateX=True, updateY=True)
                        direction = "left" if self.controller.pose["position"]["y"] < 0 else "right"
                        useSmallSteps = True if np.abs(self.controller.pose["position"]["y"]) < 0.05 else False
                        time.sleep(1e-2)
                        self.controller.turnHorizontally(direction=direction, precision=useSmallSteps)
                else:
                    self.STATE = ST_STOP
                
                # Logging
                if self.writeLogs:
                    pose = self.controller.getPose()
                    self.log(self.curTime, pose, self.STATE, handPoints, self.leverPose)


        except Exception as e:
            rospy.loginfo(f"Exception occured!\n{str(e)}")
            self.handTracker.endStream() # Remember to end the stream
            self.controller.endController()
            if thread is not None:
                thread.quit()

    def _leverPoseCallback(self, msg):
        self.leverPose = msg

    def setWristThreshold(self, threshold):
        '''
        In:
            threshold: (2x1) Array(float)
        '''
        self.hm.setWristThreshold(threshold)

    def setFingerThreshold(self, threshold):
        '''
        In:
            threshold: (float)
        '''
        self.hm.setFingerThreshold(threshold)

    def setThumbThreshold(self, threshold):
        '''
        In:
            threshold: (2x1) Array(float)
        '''
        self.hm.setThumbThreshold(threshold)

    def setCurrentGoal(self, goal):
        '''
        In:
            goal: (Float) radians
        '''
        self.agent.setGoal(goal)

    def getCurrentImage(self):
        '''
        Out:
            (2x1) (1080x1920x3) Array(float)
        '''
        return self.imgLM

    def log(self, t=None, pose=None, state=None, handPoints=None, leverPose=None):
        self.logRobotPose(pose)
        self.logFSMState(state)
        self.logHandPoints(handPoints)
        self.logLeverPose(leverPose)
        self.logTimeSteps(t)

    def logRobotPose(self, pose):
        with open(f"{self.pathToLogPose}", "a") as fp:
            writer = csv.writer(fp)

            if pose is None:
                writer.writerow(["None"])
                return

            formattedPose = []
            for key, value in pose.items():
                formattedPose.append(key)
                for _, val2 in value.items():
                    formattedPose.append(val2)
            writer.writerow(formattedPose)

    def logFSMState(self, state):
        with open(f"{self.pathToLogFSMState}", "a") as fp:
            writer = csv.writer(fp)

            if state is None:
                writer.writerow(["None"])
                return
            
            writer.writerow([state])


    def logHandPoints(self, handPoints):
        with open(f"{self.pathToLogHandPoints}", "a") as fp:
            writer = csv.writer(fp)

            if handPoints == {}:
                writer.writerow(["None"])
                return
            
            formattedHP = []
            for i in range(len(handPoints[0])):
                formattedHP.append(i)
                for _, val in handPoints[0][i].items():
                    formattedHP.append(val)
            
            writer.writerow(formattedHP)
    
    def logLeverPose(self, leverPose):
        with open(f"{self.pathToLogLeverPose}", "a") as fp:
            writer = csv.writer(fp)

            if leverPose is None:
                writer.writerow(["None"])
                return

            formattedPose = []
            formattedPose.append([leverPose.estimated_angle,
                                  leverPose.measured_angle,
                                  [leverPose.estimated_position],
                                  [leverPose.measured_position]])
            writer.writerow(formattedPose)
    
    def logTimeSteps(self, t):
        with open(f"{self.pathToLogTimeSteps}", "a") as fp:
            writer = csv.writer(fp)

            if t is None:
                writer.writerow(["None"])
                return
            
            writer.writerow([str(t)])

    def logRewardAndSuccess(self, t, reward, success, currentGoal):
        rospy.loginfo(f"Logging reward and success at path {self.pathToLogRewardSuccess}")
        with open(f"{self.pathToLogRewardSuccess}", "a") as fp:
            writer = csv.writer(fp)
            writer.writerow([t, reward, success, currentGoal])


if __name__ == "__main__":
    fsm = FSM()
    fsm.run()