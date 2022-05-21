#!/usr/bin/env python
import sys, os
import rospy
import json
import socket
import time
import numpy as np
from mimir.msg import LeverPose
from lever_detection.lever_pose_estimator import LeverPoseEstimator
from mimir.Tyr.Comms.RealSenseCam import CameraStream

def leverAnglePublisher(ip, port, cam_sn, camera_matrix, show_stream=False):
    pub = rospy.Publisher('mimir/lever_angle_pose', LeverPose)
    rospy.init_node('lever_angle_publisher', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    lever_pose = LeverPose()
    lever_pose.measured_position = [26e-2, 0., 4e-2] # [23e-2, 0., 12.9e-2]

    lever_pose_est = LeverPoseEstimator(camera_matrix)

    # Setup a connection to the RealSense camera
    camera_stream = CameraStream(cam_sn)
    camera_stream.start()

    # Setup UDP connection
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # UDP
    sock.bind((ip, port))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)

    try:
        while not rospy.is_shutdown():
            data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
            measured_angle = float(data)

            # print(f"Measured angle: {np.rad2deg(measured_angle):.2f}\t Data: {data}")
            
            try:
                angle_est, pos_est = lever_pose_est.estimatePoseFromCamStream(camera_stream, show_image=show_stream)
            except:
                angle_est = 0.
                pos_est = [0., 0., 0.]

            lever_pose.estimated_angle = angle_est
            lever_pose.estimated_position = list(pos_est)
            lever_pose.estimated_position[0] += 3e-2 # Add offset to x-position
            lever_pose.measured_angle = measured_angle

            # log_string = (f"\nEstimated angle: {np.rad2deg(lever_pose.estimated_angle):.2f}"
            #               f"\nMeasured angle: {np.rad2deg(lever_pose.measured_angle):.2f}"
            #               f"\nEstimated position: {lever_pose.estimated_position}"
            #               f"\nMeasured position: {lever_pose.measured_position}")
            # rospy.loginfo(log_string)
            if lever_pose is not None:
                pub.publish(lever_pose)
            rate.sleep()
    except KeyboardInterrupt:
        camera_stream.stop()
        sock.close()

if __name__ == '__main__':
    # f = open("/home/simon/catkin_ws/src/mimir/src/mimir/Tyr/settings.json")
    # settings = json.load(f)

    # udp_ip = settings["udp_ip"]
    # udp_port = settings["udp_port"]
    # cam_sn = settings["eff_cam_sn"]

    # fx = 954.318
    # fy = 954.318
    # u = 961.893
    # v = 534.266

    udp_ip = rospy.get_param("/mimir/udp_ip")
    udp_port = rospy.get_param("/mimir/udp_port")
    cam_sn = rospy.get_param("/mimir/cam_sn")
    fx = rospy.get_param("/mimir/fx")
    fy = rospy.get_param("/mimir/fy")
    u = rospy.get_param("/mimir/u")
    v = rospy.get_param("/mimir/v")

    camera_matrix = np.array([[fx, 0., u],
                              [0., fy, v],
                              [0., 0., 1.]])


    try:
        leverAnglePublisher(udp_ip, udp_port, cam_sn, camera_matrix, show_stream=True)
    except rospy.ROSInterruptException:
        pass