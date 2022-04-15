import cv2
import os, sys
import matplotlib.pyplot as plt
import numpy as np

def calculateArucoPose(corners, object_points, camera_matrix):
    ret, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, None)
    # rvec is the rotation vector from world coordinates to camera coordinates [x ang, y ang z ang]
    return rvec, tvec

def detectArucoFromImg(image):
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

def visualizeArucoCorners(image, corners):
    '''
    Visualize aruco corners
    In:
        image: image in rgb format to visualize aruco corners
        corners: corners of aruco marker
    '''
    plt.imshow(image)
    plt.scatter(corners[:,0], corners[:,1])
    plt.show()

def getProjectionAngle(corners):
    x1 = corners[0,0] - corners[3,0]
    y1 = -(corners[0,1] - corners[3,1])
    angle1 = np.arctan2(x1, y1)

    x2 = corners[1,0] - corners[2,0]
    y2 = -(corners[1,1] - corners[2,1])
    angle2 = np.arctan2(x2, y2)

    projection_angle_est = (angle1 + angle2) / 2
    return projection_angle_est



if __name__ == "__main__":

    path_to_img = f"{os.path.dirname(sys.argv[0])}/imgs/13_Color.png"

    image = cv2.imread(path_to_img) # load image as bgr
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to rgb
    corners, ids = detectArucoFromImg(image)
    if not (corners is () or ids is None):
        if ids.__len__() > 1:
            temp = []
            for i in range(ids.__len__()):
                temp.append(corners[i][0])
        corners = np.vstack(temp)
        visualizeArucoCorners(image, corners)
    else:
        print("No ArUco detected")

# Object points on the form [x, y, z]
    # object_points = np.array([[0., 2.45e-2, 2.45-2],
    #                           [0., 0., 2.45-2],
    #                           [0., 0., 0.],
    #                           [0., 2.45e-2, 0.]])
    # # f_len = 1.88e-3
    # fx = 954.318
    # fy = 954.318
    # u = 961.893
    # v = 534.266
    # camera_matrix = np.array([[fx, 0., u],
    #                           [0., fy, v],
    #                           [0., 0., 1.]])
    # calculateArucoPose(corners[0][0], object_points, camera_matrix)