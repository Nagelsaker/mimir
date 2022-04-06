# import rclpy
import rospy
# from rclpy.node import Node
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import JointState
from open_manipulator_msgs.msg import KinematicsPose, JointPosition
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest, SetJointPosition, SetJointPositionRequest



class SetPositionClient():
    '''
    SetPositionClient class

    Client node which requests a position for the manipulators end-effector from the
    set_position_node server node
    '''
    def __init__(self):
        rospy.wait_for_service("/goal_joint_space_path_to_kinematics_position")
        self.client = rospy.ServiceProxy('/goal_joint_space_path_to_kinematics_position', SetKinematicsPose)
        self.req = SetKinematicsPoseRequest()

    def sendRequest(self, goalPose={"position": {"x" : 0.1, "y" : 0.0, "z" : 0.22},
                                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w" : 1.0}},
                                pathTime=1.5):
        '''
        Function which sends a position request to the manipulator
        '''
        
        pose = Pose()

        point = Point()
        point.x = goalPose["position"]["x"]
        point.y = goalPose["position"]["y"]
        point.z = goalPose["position"]["z"]
        pose.position = point

        quaternion = Quaternion()
        quaternion.x = goalPose["orientation"]["x"]
        quaternion.y = goalPose["orientation"]["y"]
        quaternion.w = goalPose["orientation"]["z"]
        quaternion.z = goalPose["orientation"]["w"]
        pose.orientation = quaternion
        
        kinematics_pose = KinematicsPose()
        kinematics_pose.pose = pose
        kinematics_pose.max_accelerations_scaling_factor = 0.0
        kinematics_pose.max_velocity_scaling_factor = 0.0
        kinematics_pose.tolerance = 0.0

        self.req.planning_group = ""
        self.req.end_effector_name = "gripper"
        self.req.kinematics_pose = kinematics_pose
        self.req.path_time = pathTime

        self.client(self.req)


class SetOrientationClient():
    '''
    SetOrientationClient class

    Client node which communicates with the Robotis OpenManipulator through
    the ROS2 framework. Requests an orientation of the end-effector from the
    set_orientation_node server node
    '''
    def __init__(self):
        # super().__init__('set_orientation_node')
        rospy.wait_for_service("/goal_joint_space_path_to_kinematics_orientation")
        self.client = rospy.ServiceProxy('/goal_joint_space_path_to_kinematics_orientation', SetKinematicsPose)
        self.req = SetKinematicsPoseRequest()

    def sendRequest(self, goalPose={"position": {"x" : 0.1, "y" : 0.0, "z" : 0.22},
                                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w" : 1.0}},
                                pathTime=1.5):
        '''
        Function which requests an orientation of the end-effector
        '''
        
        pose = Pose()

        point = Point()
        pose.position = point

        quaternion = Quaternion()
        quaternion.x = goalPose["orientation"]["x"]
        quaternion.y = goalPose["orientation"]["y"]
        quaternion.z = goalPose["orientation"]["z"]
        quaternion.w = goalPose["orientation"]["w"]
        pose.orientation = quaternion
        
        kinematics_pose = KinematicsPose()
        kinematics_pose.pose = pose
        kinematics_pose.max_accelerations_scaling_factor = 0.0
        kinematics_pose.max_velocity_scaling_factor = 0.0
        kinematics_pose.tolerance = 0.0

        self.req.planning_group = ""
        self.req.end_effector_name = "gripper"
        self.req.kinematics_pose = kinematics_pose
        self.req.path_time = pathTime

        self.client(self.req)


class SetJointPositionClient():
    '''
    SetJointPositionClient class

    Client node which requests position for the fourth joint from the
    set_joint_position_node server node
    '''
    def __init__(self):
        rospy.wait_for_service("/goal_joint_space_path")
        self.client = rospy.ServiceProxy('/goal_joint_space_path', SetJointPosition)
        self.req = SetJointPositionRequest()

    def sendRequest(self, position, pathTime=1.5):
        '''
        Function which requests fourth joint position

        Assuming joint = joint4

        Joint positions are given in radians
        '''

        # TODO: Magick numbers
        minVal = -1.80
        maxVal = 2.10

        if not minVal <= position["joint4"] <= maxVal:
            if position["joint4"] < minVal:
                position["joint4"] = minVal
            else:
                position["joint4"] = maxVal

        jointPosition = JointPosition()
        jointPosition.joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']
        jointPosition.position =  [position[key] for key in position]
        jointPosition.max_accelerations_scaling_factor = 0.0
        jointPosition.max_velocity_scaling_factor = 0.0

        self.req.planning_group = ""
        self.req.joint_position = jointPosition
        self.req.path_time = pathTime

        self.client(self.req)

class SetGripperDistanceClient():
    '''
    SetGripperDistanceClient class

    Client node which requests gripper distance from the set_gripper_node server node

    MSG: open_manipulator_msgs/JointPosition
    string[]   joint_name
    float64[]  position
    float64    max_accelerations_scaling_factor
    float64    max_velocity_scaling_factor
    '''

    def __init__(self):
        rospy.wait_for_service('/goal_tool_control')
        self.client = rospy.ServiceProxy('/goal_tool_control', SetJointPosition)
        self.req = SetJointPositionRequest()

    def sendRequest(self, position, pathTime=0.4):
        '''
        In:
            position: (Float)
                Valid numbers between -0.010 and 0.010 (m)
            pathTime: (Float)
        '''
        # TODO: Magic numbers
        minVal = -0.010
        maxVal = 0.010

        if not minVal <= position <= maxVal:
            if position < minVal:
                position = minVal
            else:
                position = maxVal

        jointPosition = JointPosition()
        jointPosition.joint_name = ["gripper"]
        jointPosition.position =  [position]
        jointPosition.max_accelerations_scaling_factor = 0.0
        jointPosition.max_velocity_scaling_factor = 0.0

        self.req.planning_group = ""
        self.req.joint_position = jointPosition
        self.req.path_time = pathTime

        self.client(self.req)


class PoseSubscriber(rospy.Subscriber):
    '''
    PoseSubscriber class

    Subscriber which listens to the kinematics_pose topic
    '''
    def __init__(self):
        super().__init__('gripper/kinematics_pose', KinematicsPose, self.listener_callback)

        self.pose={"position": {"x" : 0.1, "y" : 0.0, "z" : 0.0},
                                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w" : 1.0}}


    def listener_callback(self, msg):
        '''
        In:
            msg: 
        '''
        self.pose["position"]["x"] = msg.pose.position.x
        self.pose["position"]["y"] = msg.pose.position.y
        self.pose["position"]["z"] = msg.pose.position.z
        self.pose["orientation"]["x"] = msg.pose.orientation.x
        self.pose["orientation"]["y"] = msg.pose.orientation.y
        self.pose["orientation"]["z"] = msg.pose.orientation.z
        self.pose["orientation"]["w"] = msg.pose.orientation.w
        # print(self.pose)
    
    def getPose(self):
        return self.pose




class JointPositionSubscriber(rospy.Subscriber):
    '''
    JointPositionSubscriber class

    Subscriber which listens to the joint_states topic
    '''
    def __init__(self):
        super().__init__('joint_states', JointState, self.listener_callback)
        self.joints = {}

    def listener_callback(self, msg):
        '''
        In:
            msg: 
        '''
        self.joints["joint1"] = msg.position[0]
        self.joints["joint2"] = msg.position[1]
        self.joints["joint3"] = msg.position[2]
        self.joints["joint4"] = msg.position[3]
        self.joints["gripper"] = msg.position[4]

    def getPositions(self):
        return self.joints


def main(args=None):
    # rclpy.init(args=args)

    jointPosSub = JointPositionSubscriber()
    set_jp_client = SetJointPositionClient()
    jp = {"joint1" : 0.0, "joint2" : -1.0, "joint3" : 0.0, "joint4" : 1.0, "gripper" : 0.01}
    set_jp_client.sendRequest(jp)

    so_client = SetPositionClient()
    so_client.sendRequest()
    while(True):
        j1 = jointPosSub.getPositions()["joint1"]
        j2 = jointPosSub.getPositions()["joint2"]
        j3 = jointPosSub.getPositions()["joint3"]
        j4 = jointPosSub.getPositions()["joint4"]
        gr = jointPosSub.getPositions()["gripper"]
        print(f"j1: {j1:.2f}, j2: {j2:.2f}, j3: {j3:.2f}, j4: {j4:.2f}, gr: {gr:.2f},  \r", end="")


if __name__ == '__main__':
    rospy.init_node('test_testing')
    main()