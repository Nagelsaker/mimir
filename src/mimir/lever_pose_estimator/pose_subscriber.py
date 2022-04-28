#!/usr/bin/env python
import rospy
import numpy as np
from mimir.msg import LeverPose
# from lever_pose import LeverPose

def callback(data):
    # log_string = (f"\nEstimated angle: {np.rad2deg(data.estimated_angle):.2f}"
    #               f"\nMeasured angle: {np.rad2deg(data.measured_angle):.2f}"
    #               f"\nEstimated position: {data.estimated_position}"
    #               f"\nMeasured position: {data.measured_position}")
    # rospy.loginfo(log_string)
    pass
    
def listener():
    rospy.init_node('lever_angle_subscriber', anonymous=True)

    rospy.Subscriber("mimir/lever_angle_pose", LeverPose, callback)

    rospy.spin()

if __name__ == '__main__':
    listener()