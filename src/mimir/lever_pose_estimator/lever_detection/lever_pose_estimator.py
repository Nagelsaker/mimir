#!/usr/bin/env python

from dis import show_code
import cv2
import os
import sys
import signal
import rospy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

from sensor_msgs.msg import JointState
from open_manipulator_rl_environments.helper_functions.forward_kinematics import OMXForwardKinematics
from mimir.Tyr.Comms.RealSenseCam import CameraStream

OBJECT_POINTS_ALL = {   
                        0 : np.array([[0, 2.45e-2, 7.67e-2],
                                      [0, 0, 7.67e-2],
                                      [0, 0, 5.2e-2],
                                      [0, 2.45e-2, 5.2e-2]], np.float32),
                        1 : np.array([[0, 2.45e-2, 5.07e-2],
                                      [0, 0, 5.07e-2],
                                      [0, 0, 2.6e-2],
                                      [0, 2.45e-2, 2.6e-2]], np.float32),
                        3 : np.array([[0, 2.45e-2, 2.45e-2],
                                        [0, 0, 2.45e-2],
                                        [0, 0, 0],
                                        [0, 2.45e-2, 0]], np.float32)
                    }

# CAMERA_DHPARAM_MEASUREMENT = {"T1" : [0, 3.25e-2, -6.7e-2, np.pi/2],
#                               "T2" : [0., 7.6e-2, 0., 0.]}
CAMERA_DHPARAM_MEASUREMENT = {"T1" : [0, 6.7e-2, 3.25e-2, -np.pi/2],
                              "T2" : [-np.pi/2, 7.6e-2, 0., np.pi/2]}


JOINT_SPACE_SUBSCRIBER = '/joint_states'

class LeverPoseEstimator(object):
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        self.all_obj_p = OBJECT_POINTS_ALL
        self.T1_params = CAMERA_DHPARAM_MEASUREMENT["T1"]
        self.T2_params = CAMERA_DHPARAM_MEASUREMENT["T2"]
        self._calculateCamToEffTransform()

        # Subscribe to the OMX joint states
        self.joint_space_sub = rospy.Subscriber(JOINT_SPACE_SUBSCRIBER, JointState, self.omx_joints_callback)
        self.joint_space_values = JointState()
        self.joint_states = np.zeros(6)

        self.omx_kinematics = OMXForwardKinematics()

        self.previous_angle_est = 0
        self.previous_pos_est = [0, 0, 0]

    def estimatePoseFromCamStream(self, camera_stream, show_image=False):
            image,_ = camera_stream.getImages()
            corners, ids = self.detectArucoFromImg(image)

            stacked_corners, obj_p = self._getRelatedObjAndCornerPnts(corners, ids)
            if show_image:
                image_corners = self.drawPointsOnImage(image, stacked_corners)
                cv2.namedWindow('omx stream', cv2.WINDOW_NORMAL)
                cv2.imshow('omx stream', image_corners)
                cv2.waitKey(1)

            angle, lp = self.getLeverPoseEstimate(corners, ids)

            return angle, lp

    def detectArucoFromImg(self, image):
        '''
        Detect aruco marker from image
        In:
            image: image in rgb format to detect aruco marker
        
        Out:
            corners: corners of aruco marker
        '''

        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
        return corners, ids
    

    def omx_joints_callback(self, data):
        '''
        Topic callback for OM-X joint states
        '''
        self.joint_space_values = data
    
    def _getJointStates(self):
        '''
        Return the OM-X joints states

        In Gazebo the states is in the following order:
            [gripper, gripper_sub, joint1, joint2, joint3, joint4] Length=6
        While in real-world the order is:
            [joint1, joint2, joint3, joint4, gripper] Length=5
        
        The Joints states defined here will have the following order:
            [gripper, gripper_sub joint1, joint2, joint3, joint4] Length=6
        '''
        
        if self.joint_space_values.name != []:
            temp = self.joint_space_values.position

            if len(temp) == 6:
                self.joint_states = temp
            elif len(temp) == 5:
                self.joint_states = np.hstack([temp[-1], temp[-1], temp[:-1]])

        return self.joint_states


        
    def getLeverPoseEstimate(self, corners, ids, H_eff_w=None):
        '''
        This function should take in the projection angle and end effector pose,
        and return an estimate of the lever angle
        In:
            projection_angle: angle between the projection of the aruco marker and the
                camera's optical axis
            eff_pose: (x,y,z) position and (x, y , z, w) quaternion of the manipulator's end effector
        Out:
            lever_angle: estimate of the lever angle
        '''
        if ids is None:
            # No aruco marker detected
            angle_est = self.previous_angle_est
            return angle_est, self.previous_pos_est
        elif len(ids) == 3:
            left_ids = [0, 3, 4, 7, 8, 11]
            right_ids = [1, 2, 5, 6, 9, 10]
        elif len(ids) == 2:
            left_ids = [0, 3, 4, 7]
            right_ids = [1, 2, 5, 6]
        elif len(ids) == 1:
            left_ids = [0, 3]
            right_ids = [1, 2]
        stacked_corners, visible_obj_p = self._getRelatedObjAndCornerPnts(corners, ids)
        rvec, tvec = self._getRotAndTransVec(stacked_corners, visible_obj_p)

        H_targ_cam = np.zeros((4,4))
        H_targ_cam[:3,:3] = cv2.Rodrigues(rvec)[0]
        H_targ_cam[:3,3] = tvec.T
        H_targ_cam[3,3] = 1

        H_targ_cam_inv = np.linalg.inv(H_targ_cam)

        H_cam_eff = self.getCamToEffTransform()

        self.omx_kinematics.setDHParams(self._getJointStates()[2:])
        H_eff_w = self.omx_kinematics.getTransformFromEffToWorld()

        H_targ_w = H_eff_w @ H_cam_eff @ H_targ_cam

        obj_p_hom = self._vectorToHomogeneous(visible_obj_p)
        object_coords_w = self._vecToInhomogeneous((H_targ_w @ obj_p_hom.T).T)


        # TODO: Verify the sign of the angles
        # Interpolate left line
        # We do not care about y values, since the angle is in the XZ plane
        left_points = np.vstack([object_coords_w[left_ids][:,0], object_coords_w[left_ids][:,2]]).T
        # left_line = interp1d(left_points[:,0], left_points[:,1])
        # left_angle = np.arctan2(left_line(left_points[0,0]), left_line(left_points[-1,0]))
        left_angle = np.arctan2((left_points[-1,0] - left_points[0,0]), np.abs(left_points[0,1] - left_points[-1,1]))

        # Interpolate right line
        right_points = np.vstack([object_coords_w[right_ids][:,0], object_coords_w[right_ids][:,2]]).T
        # right_line = interp1d(right_points[:,0], right_points[:,1])
        # right_angle = np.arctan2(right_line(right_points[0,0]), right_line(right_points[-1,0]))
        right_angle = np.arctan2((right_points[-1,0] - right_points[0,0]), np.abs(right_points[0,1] - right_points[-1,1]))

        angle_est = (left_angle + right_angle)/2
        self.previous_angle_est = angle_est

        # Estimate lever position
        lever_pos_obj = np.array([-1e-2, -2e-2, -10e-2])
        lever_pos_w = self._vecToInhomogeneous(H_targ_w @ self._vectorToHomogeneous(lever_pos_obj))
        self.previous_pos_est = lever_pos_w

        # Plot object_coords_w in a 3d scatter plot
        # self._plot3d(object_coords_w)
        return angle_est, lever_pos_w


    def getCamToEffTransform(self):
        return self.H_cam_eff


    def _getRotAndTransVec(self, stacked_corners, visible_obj_p):
        ret, rvec, tvec = cv2.solvePnP(visible_obj_p, stacked_corners, self.camera_matrix, None)
        # rvec is the rotation vector from world coordinates to camera coordinates [x ang, y ang z ang]
        return rvec, tvec

    def _getRelatedObjAndCornerPnts(self, corners, ids):
        '''
        Stacks corner points into one array, and creates an array of visible object points
        based on the detected aruco markers
        In:
            corners: (n,2) array, corners of aruco marker
            ids: (n, 1) array, ids of aruco marker
        Out:
            stacked_corners: (n,2) array of corner points
            visible_obj_p: (n,3) array of object points
        '''
        if ids is None:
            return -1, -1
        n = len(ids)
        visible_obj_p = np.zeros((n*4, 3), np.float32)
        stacked_corners = np.zeros((n*4, 2), np.float32)

        i = 0
        for id in ids[:,0]:
            visible_obj_p[i*4:i*4+4,:] = self.all_obj_p[id]
            stacked_corners[i*4:i*4+4,:] = corners[i]
            i += 1

        return stacked_corners, visible_obj_p


    def _calculateCamToEffTransform(self):
        # H_c1_eff = self._makeTransformFromDHParams(theta=self.T1_params[0],
        #                                           d=self.T1_params[1],
        #                                           a=self.T1_params[2],
        #                                           alpha=self.T1_params[3])
        # H_cam_c1 = self._makeTransformFromDHParams(theta=self.T2_params[0],
        #                                           d=self.T2_params[1],
        #                                           a=self.T2_params[2],
        #                                           alpha=self.T2_params[3])

        H_c1_cam = self._makeTransformFromDHParams(theta=self.T1_params[0],
                                                  d=self.T1_params[1],
                                                  a=self.T1_params[2],
                                                  alpha=self.T1_params[3])
        
        H_cam_c1 = np.linalg.inv(H_c1_cam)

        H_eff_c1 = self._makeTransformFromDHParams(theta=self.T2_params[0],
                                                  d=self.T2_params[1],
                                                  a=self.T2_params[2],
                                                  alpha=self.T2_params[3])
        H_c1_eff = np.linalg.inv(H_eff_c1)

        H_cam_eff = H_c1_eff @ H_cam_c1 

        self.H_cam_eff = H_cam_eff


    def _makeTransformFromDHParams(self, theta, d, a, alpha):
        '''
        In:
            theta: Rot angle in rad around z-axis
            d: Displacement along z-axis
            a: displacement along x-axis
            alpha: Rot angle in rad around x-axis
        '''
        rot_z = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                        [np.sin(theta), np.cos(theta), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        T_z = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, d],
                        [0, 0, 0, 1]])
        T_a = np.array([[1, 0, 0, a],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        rot_x = np.array([[1, 0, 0, 0],
                        [0, np.cos(alpha), -np.sin(alpha), 0],
                        [0, np.sin(alpha), np.cos(alpha), 0],
                        [0, 0, 0, 1]])
        T = rot_z@T_z@T_a@rot_x
        return T
    
    def _vectorToHomogeneous(self, vector):
        '''
        In:
            vector: (n,3) array
        Out:
            vector: (n,4) array
        '''
        if len(vector.shape) == 1:
            n = 1
            m = vector.shape[0]
        elif len(vector.shape) == 2:
            n, m = vector.shape
        else:
            raise Exception("Vector must be 1 or 2 dimensional")

        if n > 1:
            vector = np.hstack((vector, np.ones((n, 1))))
        else:
            vector = np.hstack((vector, 1))
        return vector
    
    def _vecToInhomogeneous(self, vector):
        '''
        In:
            vector: (n,4) array / (n,3) array
        Out:
            vector: (n,3) array
        '''
        if len(vector.shape) == 2:
            n = vector.shape[0]
            m = vector.shape[1]
        else:
            n = 1
            m = vector.shape[0]
        if m == 4:
            if n > 1:
                out = np.vstack([vec[:3] / vec[3] for vec in vector])
            else:
                out = vector[:3] / vector[3]
        elif m == 3:
            if n > 1:
                out = np.vstack([vec[:2] / vec[2] for vec in vector])
            else:
                out = vector[:2] / vector[2]
        else:
            raise Exception('Vector has wrong shape')
        return out


    def _plot3d(self, object_coords_w):
        '''
        Plots object coordinates given in world frame in a 3D scatter plot
        using Matplotlib

        In:
            object_coords_w: (n,3) array of object coordinates in world frame
        '''
        fig = fig = plt.figure(figsize=plt.figaspect(1)*1.5)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(object_coords_w[1:,0], object_coords_w[1:,1], object_coords_w[1:,2])
        ax.scatter(object_coords_w[0,0], object_coords_w[0,1], object_coords_w[0,2], c='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()


    def visualizeArucoCorners(self, image, corners):
        '''
        Visualize aruco corners
        In:
            image: image in rgb format to visualize aruco corners
            corners: corners of aruco marker
        '''
        plt.imshow(image)
        plt.scatter(corners[:,0], corners[:,1])
        plt.show()

    def drawPointsOnImage(self, image, points):
        '''
        Draw 2D points on an image and return the new image
        In:
            image: (h,w,3) array of image
            points: (n,2) array of points
        Out:
            image: (h,w,3) array of image with points drawn on it
        '''
        if type(points) == int:
            return image
        image = image.copy()
        for point in points:
            cv2.circle(image, (int(point[0]), int(point[1])), 8, (0,0,255), -1)
        return image


if __name__ == "__main__":
    rospy.init_node("lever_angle_estimator")
    print("Hey hey")
    path_to_img = f"{os.path.dirname(sys.argv[0])}/imgs/14_Color.png"
    image = cv2.imread(path_to_img) # load image as bgr
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to rgb
    fx = 954.318
    fy = 954.318
    u = 961.893
    v = 534.266
    camera_matrix = np.array([[fx, 0., u],
                              [0., fy, v],
                              [0., 0., 1.]])

    lp_est = LeverPoseEstimator(camera_matrix)

    # Setup a connection to the RealSense camera
    cam_sn = "838212074315"
    camera_stream = CameraStream(cam_sn)
    camera_stream.start()

    while True:
        try:
            image,_ = camera_stream.getImages()
            corners, ids = lp_est.detectArucoFromImg(image)

            stacked_corners, obj_p = lp_est._getRelatedObjAndCornerPnts(corners, ids)
            image_corners = lp_est.drawPointsOnImage(image, stacked_corners)
            cv2.namedWindow('omx stream', cv2.WINDOW_NORMAL)
            cv2.imshow('omx stream', image_corners)
            cv2.waitKey(1)

            angle, lp = lp_est.getLeverPoseEstimate(corners, ids)
            print(f"Angle estimate1: {np.rad2deg(angle)}\tLever pos est: {lp}         \r", end="")

        except KeyboardInterrupt:
            camera_stream.stop()
            break
